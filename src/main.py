# Essential modules
import argparse
import torch
import wandb
import os
import random
import copy

# Metric
from calibration.ece import init_config
from hessian import calculator

# Training
from model import build_model, ModelSetting
from opt import (
    build_optimizer,
    OptimizerSetting,
    linear_warmup_cosine_decay,
    requires_previous_model,
)
from loss import FocalLoss
from algorithm.base_algorithm import get_algorithm_class
from algorithm.algo_erm import ERM

# Utils
from utils import misc
from utils import calc

# Dataset
from dataset.dataset import build_dataset, DatasetSetting
from dataset.dataloader import eval_loader
from dataset.preprocess.imagenet_v2 import variants
from dataset.preprocess.cifar10_c import corruptions


def build_wandb_init_kwargs(config, project, name, entity):
    kwargs = {
        "config": config,
        "project": project,
        "name": name,
    }
    if entity:
        kwargs["entity"] = entity
    return kwargs


# Main Loop
def train_loop(exp_dict, ece_config):

    # ========================= Setting =========================
    device = exp_dict['device']
    misc.set_seeds(exp_dict['seed'])

    # ========================= Model =========================
    model = build_model(ModelSetting(name=exp_dict['model'], 
                                     num_classes=exp_dict['num_classes'],
                                     num_dim=exp_dict['num_dim']))
    model.to(device)
    misc.print_model_summary(model, exp_dict['dataset'])

    # AD-SGD
    if requires_previous_model(exp_dict['opt']):
        prev_model = copy.deepcopy(model)
        prev_model.to(device)
    else:
        prev_model = None
    
    if exp_dict['num_gpu'] > 1:
        print("DataParallel")
        model = torch.nn.DataParallel(model)

        if requires_previous_model(exp_dict['opt']):
            prev_model = torch.nn.DataParallel(prev_model)

    # ========================= Dataset =========================
    dataset = build_dataset(DatasetSetting(name=exp_dict['dataset'], 
                                           root=exp_dict['data_root']))

    # Train Loader
    # Ref https://pytorch.org/docs/stable/notes/randomness.html
    fixed_generator_train = torch.Generator()
    fixed_generator_train.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(dataset.train_dataset,
                                              batch_size=exp_dict['batch_size'],
                                              shuffle=True,
                                              num_workers=exp_dict['num_workers'],
                                              worker_init_fn=misc.seed_worker,
                                              generator=fixed_generator_train)

    # Val Loader
    fixed_generator_val = torch.Generator()
    fixed_generator_val.manual_seed(0)
    val_loader = torch.utils.data.DataLoader(dataset.val_dataset,
                                             batch_size=exp_dict['batch_size'],
                                             shuffle=False,
                                             num_workers=exp_dict['num_workers'],
                                             worker_init_fn=misc.seed_worker,
                                             generator=fixed_generator_val)

    # Test Loader
    ood_dataloaders = []
    kinds_of_ood = []
    
    ## ImageNet -> ImageNetV2                                    
    if exp_dict['ood_dataset'] == 'imagenet_v2':
        kinds_of_ood = variants
        for kind in kinds_of_ood:
            tmp_loader = eval_loader(exp_dict, kind)
            ood_dataloaders.append(tmp_loader)

    ## CIFAR10 -> CIFAR10_C
    elif exp_dict['ood_dataset'] == 'cifar10_c':
        kinds_of_ood = corruptions
        for kind in kinds_of_ood:
            tmp_loader = eval_loader(exp_dict, kind)
            ood_dataloaders.append(tmp_loader)

    ## CIFAR10 -> CIFAR10_1
    elif exp_dict['ood_dataset'] == 'cifar10_1': 
        kinds_of_ood = ['test']
        test_loader = eval_loader(exp_dict, None)
        ood_dataloaders.append(test_loader)

    else:
        print(f"[OOD Dataset: {exp_dict['ood_dataset']}] is Not Supported")

    # ========================= Optimizer =========================
    opt = build_optimizer(OptimizerSetting(name=exp_dict['opt'],
                                           lr=exp_dict['lr'],
                                           weight_decay=exp_dict['weight_decay'],
                                           model=model,
                                           momentum=exp_dict["momentum"],
                                           eps=exp_dict["eps"],
                                           beta_1=exp_dict["beta_1"],
                                           beta_2=exp_dict["beta_2"],
                                           rho=exp_dict["rho"],
                                           amplifier=exp_dict["amplifier"]
                                           ))
    
    if requires_previous_model(exp_dict['opt']):
        prev_opt = build_optimizer(OptimizerSetting(name=exp_dict['opt'],
                                                    lr=exp_dict['lr'],
                                                    weight_decay=exp_dict['weight_decay'],
                                                    model=prev_model,
                                                    momentum=exp_dict["momentum"],
                                                    eps=exp_dict["eps"],
                                                    beta_1=exp_dict["beta_1"],
                                                    beta_2=exp_dict["beta_2"],
                                                    rho=exp_dict["rho"],
                                                    amplifier=exp_dict["amplifier"]
                                                    ))
    else:
        prev_opt = None
    
    # ========================= Loss Function =========================
    if exp_dict['loss_function'] == 'focal_loss':
        loss_function = FocalLoss(gamma=exp_dict['focal_gamma'], size_average=True)
    elif exp_dict['loss_function'] == 'cross_entropy_loss':
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    eval_loss_function = torch.nn.CrossEntropyLoss()


    # ========================= Learning Rate Scheduler =========================
    if exp_dict["use_scheduler"]:
        warmup_steps = len(train_loader) * exp_dict["warmup_epochs"]
        total_steps = len(train_loader) * exp_dict["epochs_budget"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                        opt,
                        linear_warmup_cosine_decay(warmup_steps, total_steps),
                    )
    else:
        scheduler = None


    # ========================= Train and Evaluation Loop =========================
    algorithm_name = exp_dict["algorithm"]
    algorithm_class = get_algorithm_class(algorithm_name)
    algorithm = algorithm_class(model, prev_model,
                                opt, prev_opt, 
                                loss_function, scheduler, 
                                ece_config, exp_dict)
    algorithm.to(device)


    for epoch in range(0, exp_dict['epochs_budget']):

        print(f'epoch: {epoch} start')
        wandb_log_dict = {}
        wandb_log_dict['epoch'] = epoch

        ## ========================= Train =========================
        for idx, batch in enumerate(train_loader):
            algorithm.update(idx, batch)
        avg_train_loss, avg_train_acc, avg_train_ece = algorithm.epoch_end()

        wandb_log_dict['avg_train_loss'] = avg_train_loss
        wandb_log_dict['avg_train_acc'] = avg_train_acc
        wandb_log_dict['avg_train_ece'] = avg_train_ece

        ## ========================= Val =========================
        # _, avg_val_loss, avg_val_acc, avg_val_ece = val_epoch(model, None, eval_loss_function, val_loader, ece_config, exp_dict)

        for idx, batch in enumerate(val_loader):
            algorithm.predict(idx, batch)
        avg_val_loss, avg_val_acc, avg_val_ece = algorithm.epoch_end()

        wandb_log_dict['avg_val_loss'] = avg_val_loss
        wandb_log_dict['avg_val_acc'] = avg_val_acc
        wandb_log_dict['avg_val_ece'] = avg_val_ece


        ## ========================= Test =========================
        #### ImageNet -> imagenet_v2
        if ((epoch % exp_dict['ood_test_interval']) == 0) and (exp_dict['ood_dataset'] == 'imagenet_v2' or exp_dict['ood_dataset'] == 'cifar10_c' or exp_dict['ood_dataset'] == 'cifar10_1'):
            avg_test_loss, avg_test_acc, avg_test_ece = 0, 0, 0
            for test_loader, kind in zip(ood_dataloaders, kinds_of_ood):
                
                # _, tmp_test_loss, tmp_test_acc, tmp_test_ece, = test_epoch(model, None, eval_loss_function, test_loader, ece_config, exp_dict)

                for idx, batch in enumerate(test_loader):
                    algorithm.predict(idx, batch)
                tmp_test_loss, tmp_test_acc, tmp_test_ece = algorithm.epoch_end()

                avg_test_loss += tmp_test_loss
                avg_test_acc += tmp_test_acc
                avg_test_ece += tmp_test_ece
                wandb_log_dict[f'avg_loss_{kind}'] = tmp_test_loss
                wandb_log_dict[f'avg_acc_{kind}'] = tmp_test_acc
                wandb_log_dict[f'avg_ece_{kind}'] = tmp_test_ece

            num_of_dataloaders = len(ood_dataloaders)
            avg_test_loss /= num_of_dataloaders
            avg_test_acc /= num_of_dataloaders
            avg_test_ece /= num_of_dataloaders

            wandb_log_dict['avg_test_loss'] = avg_test_loss
            wandb_log_dict['avg_test_acc'] = avg_test_acc
            wandb_log_dict['avg_test_ece'] = avg_test_ece

        ##### MNIST
        else: 
            print(f"[OOD Dataset: {exp_dict['ood_dataset']}] is Not Supported")


        ## ========================= Calc Grad Norm and Hessian =========================
        if exp_dict['calc_hessian'] and ((epoch % exp_dict['calc_hessian_interval']) == 0):

            # HessianCalculator インスタンスの作成
            print("calc hessian by using pyhessian")
            hessian_calculator = calculator.HessianCalculator(model, exp_dict)
            hessian_log_dict = hessian_calculator.calc_hessian_for_datasets(train_loader, val_loader, ood_dataloaders, kinds_of_ood)
            wandb_log_dict.update(hessian_log_dict)
            
        ## Wandb Logging
        wandb.log(wandb_log_dict)
        
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibration and Out-of-Distribution Project')

    # Environmental Setting
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_gpu', type=int, default=2, help="num of gpus")

    # Experimental Setting
    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--epochs_budget', type=int, default=400)
    parser.add_argument('--model', type=str, default='resnet18_1_cifar')
    parser.add_argument('--num_dim', type=int, default=4096)

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--ood_dataset', type=str, default='cifar10_1')
    parser.add_argument('--ood_data_root', type=str, default=None)
    parser.add_argument('--ood_test_interval', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--save_model_path', type=str, default='./')

    # Optimizer and Batch 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_accumulation', type=int, default=1)
    parser.add_argument('--opt', type=str, default='vanilla_sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--use_scheduler', type=bool, default=False)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0) # follow pytorch default
    parser.add_argument('--eps', type=float, default=1e-08) # follow pytorch default
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)

    # SAM
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--esam_beta', type=float, default=0.5)
    parser.add_argument('--esam_gamma', type=float, default=1.0)

    # AD-SGD
    parser.add_argument('--amplifier', type=float, default=0.02)

    # Loss Function
    parser.add_argument('--loss_function', type=str, default='cross_entropy_loss')
    parser.add_argument("--focal_gamma", type=float, default=1, help="Gamma for focal components")

    # Stabilizer
    parser.add_argument('--use_clip_gradnorm', type=bool, default=False)

    # Hessian
    parser.add_argument('--calc_gradnorm', type=bool, default=False)
    parser.add_argument('--calc_hessian', type=bool, default=False)
    parser.add_argument('--calc_hessian_interval', type=int, default=25)
    parser.add_argument('--pyhessian_batch_size_per_gpu', type=int, default=128)
    parser.add_argument('--pyhessian_sample_size', type=int, default=1024)
    parser.add_argument('--top_n', type=int, default=1)

    # Wandb Configuration
    parser.add_argument("--wandb_exp_id", type=int, default=99999999)
    parser.add_argument("--wandb_entity", type=str, default='', help="optional wandb team or organization")
    parser.add_argument("--wandb_project_name", type=str, default='default_project', help="should include dataset and model")
    parser.add_argument('--wandb_offline', action = 'store_true')
    parser.add_argument('--debug_mode', action = 'store_true')
    parser.add_argument('--save_model', action = 'store_true')
    args = parser.parse_args()


    # ============= environmental setting =============
    device = torch.device("cuda" if (torch.cuda.is_available() and args.num_gpu > 0) else "cpu")
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    misc.set_seeds(args.seed)
    misc.print_libs_version()
    misc.define_gpus(args.dataset)
    args.data_root = misc.update_dataroot(args.dataset, args.data_root)

    # ================ wandb ================ 
    wandb_exp_name = f'exp_id-{args.wandb_exp_id}_opt-{args.opt}_bs-{args.batch_size}_lr-{args.lr}'
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(**build_wandb_init_kwargs(
        config=args,
        project=args.wandb_project_name,
        name=wandb_exp_name,
        entity=args.wandb_entity,
    ))

    print('\nWandb Setting:')
    print(f'\twandb_project_name: f{args.wandb_project_name}')
    print(f'\twandb_exp_name: f{wandb_exp_name}')


    # ============= train config =============
    exp_dict = wandb.config
    exp_dict['device'] = device

    print('\nExperimental Configuration:')
    for k, v in sorted(exp_dict.items()):
        print('\t{}: {}'.format(k, v))


    # ============= ece config =============
    ece_config = init_config()
    ece_config['num_reps'] = 100
    ece_config['norm'] = 1
    ece_config['ce_type'] = 'ew_ece_bin'
    ece_config['num_bins'] = 10

    # ================ execution ================ 
    # get trained model by using IID dataset
    trained_model = train_loop(exp_dict, ece_config)

    if args.save_model:
        save_path = f"{args.save_model_path}/{args.model}_{args.opt}_{args.lr}.pth"  # Replace with your desired path and model name
        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


        
