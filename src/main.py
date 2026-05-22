import argparse
import copy
import os
import random

import torch
import wandb

from algorithm.algo_erm import ERM
from calibration.ece import init_config
from dataset.dataloader import eval_loader
from dataset.dataset import DatasetSetting, build_dataset
from dataset.preprocess.cifar10_c import corruptions
from hessian import calculator
from model import ModelSetting, SUPPORTED_MODELS, build_model
from opt import (
    OptimizerSetting,
    SUPPORTED_OPTIMIZERS,
    build_optimizer,
    linear_warmup_cosine_decay,
    requires_previous_model,
)
from utils import misc


def build_wandb_init_kwargs(config, project, name, entity):
    kwargs = {
        "config": config,
        "project": project,
        "name": name,
    }
    if entity:
        kwargs["entity"] = entity
    return kwargs


def train_loop(exp_dict, ece_config):
    device = exp_dict["device"]
    misc.set_seeds(exp_dict["seed"])

    model = build_model(
        ModelSetting(
            name=exp_dict["model"],
            num_classes=exp_dict["num_classes"],
        )
    )
    model.to(device)
    misc.print_model_summary(model, exp_dict["dataset"])

    if requires_previous_model(exp_dict["opt"]):
        prev_model = copy.deepcopy(model)
        prev_model.to(device)
    else:
        prev_model = None

    if exp_dict["num_gpu"] > 1:
        print("DataParallel")
        model = torch.nn.DataParallel(model)
        if prev_model is not None:
            prev_model = torch.nn.DataParallel(prev_model)

    dataset = build_dataset(
        DatasetSetting(
            name=exp_dict["dataset"],
            root=exp_dict["data_root"],
        )
    )

    fixed_generator_train = torch.Generator().manual_seed(0)
    train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset,
        batch_size=exp_dict["batch_size"],
        shuffle=True,
        num_workers=exp_dict["num_workers"],
        worker_init_fn=misc.seed_worker,
        generator=fixed_generator_train,
    )

    fixed_generator_val = torch.Generator().manual_seed(0)
    val_loader = torch.utils.data.DataLoader(
        dataset.val_dataset,
        batch_size=exp_dict["batch_size"],
        shuffle=False,
        num_workers=exp_dict["num_workers"],
        worker_init_fn=misc.seed_worker,
        generator=fixed_generator_val,
    )

    ood_dataloaders = [eval_loader(exp_dict, kind) for kind in corruptions]

    opt = build_optimizer(
        OptimizerSetting(
            name=exp_dict["opt"],
            lr=exp_dict["lr"],
            weight_decay=exp_dict["weight_decay"],
            model=model,
            momentum=exp_dict["momentum"],
            eps=exp_dict["eps"],
            beta_1=exp_dict["beta_1"],
            beta_2=exp_dict["beta_2"],
            rho=exp_dict["rho"],
            amplifier=exp_dict["amplifier"],
        )
    )

    if requires_previous_model(exp_dict["opt"]):
        prev_opt = build_optimizer(
            OptimizerSetting(
                name=exp_dict["opt"],
                lr=exp_dict["lr"],
                weight_decay=exp_dict["weight_decay"],
                model=prev_model,
                momentum=exp_dict["momentum"],
                eps=exp_dict["eps"],
                beta_1=exp_dict["beta_1"],
                beta_2=exp_dict["beta_2"],
                rho=exp_dict["rho"],
                amplifier=exp_dict["amplifier"],
            )
        )
    else:
        prev_opt = None

    if exp_dict["use_scheduler"]:
        warmup_steps = len(train_loader) * exp_dict["warmup_epochs"]
        total_steps = len(train_loader) * exp_dict["epochs_budget"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            linear_warmup_cosine_decay(warmup_steps, total_steps),
        )
    else:
        scheduler = None

    algorithm = ERM(
        model,
        prev_model,
        opt,
        prev_opt,
        torch.nn.CrossEntropyLoss(),
        scheduler,
        ece_config,
        exp_dict,
    )
    algorithm.to(device)

    for epoch in range(exp_dict["epochs_budget"]):
        print(f"epoch: {epoch} start")
        wandb_log_dict = {"epoch": epoch}

        for idx, batch in enumerate(train_loader):
            algorithm.update(idx, batch)
        avg_train_loss, avg_train_acc, avg_train_ece = algorithm.epoch_end()
        wandb_log_dict["avg_train_loss"] = avg_train_loss
        wandb_log_dict["avg_train_acc"] = avg_train_acc
        wandb_log_dict["avg_train_ece"] = avg_train_ece

        for idx, batch in enumerate(val_loader):
            algorithm.predict(idx, batch)
        avg_val_loss, avg_val_acc, avg_val_ece = algorithm.epoch_end()
        wandb_log_dict["avg_val_loss"] = avg_val_loss
        wandb_log_dict["avg_val_acc"] = avg_val_acc
        wandb_log_dict["avg_val_ece"] = avg_val_ece

        if epoch % exp_dict["ood_test_interval"] == 0:
            avg_test_loss = 0.0
            avg_test_acc = 0.0
            avg_test_ece = 0.0
            for test_loader, corruption_kind in zip(ood_dataloaders, corruptions):
                for idx, batch in enumerate(test_loader):
                    algorithm.predict(idx, batch)
                tmp_test_loss, tmp_test_acc, tmp_test_ece = algorithm.epoch_end()
                avg_test_loss += tmp_test_loss
                avg_test_acc += tmp_test_acc
                avg_test_ece += tmp_test_ece
                wandb_log_dict[f"avg_loss_{corruption_kind}"] = tmp_test_loss
                wandb_log_dict[f"avg_acc_{corruption_kind}"] = tmp_test_acc
                wandb_log_dict[f"avg_ece_{corruption_kind}"] = tmp_test_ece

            num_ood_loaders = len(ood_dataloaders)
            wandb_log_dict["avg_test_loss"] = avg_test_loss / num_ood_loaders
            wandb_log_dict["avg_test_acc"] = avg_test_acc / num_ood_loaders
            wandb_log_dict["avg_test_ece"] = avg_test_ece / num_ood_loaders

        if exp_dict["calc_hessian"] and (epoch % exp_dict["calc_hessian_interval"] == 0):
            print("calc hessian by using pyhessian")
            hessian_calculator = calculator.HessianCalculator(model, exp_dict)
            hessian_log_dict = hessian_calculator.calc_hessian_for_datasets(
                train_loader,
                val_loader,
                ood_dataloaders,
                corruptions,
            )
            wandb_log_dict.update(hessian_log_dict)

        wandb.log(wandb_log_dict)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduction code for the ICLR 2024 SA-SAM paper."
    )

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of GPUs to use.")

    parser.add_argument("--epochs_budget", type=int, default=200)
    parser.add_argument("--model", type=str, default="vgg19", choices=SUPPORTED_MODELS)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--ood_dataset", type=str, default="cifar10_c", choices=["cifar10_c"])
    parser.add_argument(
        "--ood_data_root",
        type=str,
        default=None,
        help="Optional root for CIFAR10-C. Supports either the npy directory or its parent.",
    )
    parser.add_argument("--ood_test_interval", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--save_model_path", type=str, default="./")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_accumulation", type=int, default=1)
    parser.add_argument("--opt", type=str, default="sasam", choices=SUPPORTED_OPTIMIZERS)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--amplifier", type=float, default=0.02)

    parser.add_argument("--use_clip_gradnorm", action="store_true")

    parser.add_argument("--calc_gradnorm", action="store_true")
    parser.add_argument("--calc_hessian", action="store_true")
    parser.add_argument("--calc_hessian_interval", type=int, default=25)
    parser.add_argument("--pyhessian_batch_size_per_gpu", type=int, default=128)
    parser.add_argument("--pyhessian_sample_size", type=int, default=1024)
    parser.add_argument("--top_n", type=int, default=1)

    parser.add_argument("--wandb_exp_id", type=int, default=99999999)
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="Optional wandb team or organization.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="smoothness-aware-sam",
    )
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and args.num_gpu > 0) else "cpu")
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    misc.set_seeds(args.seed)
    misc.print_libs_version()
    misc.define_gpus(args.dataset)
    args.data_root = misc.update_dataroot(args.dataset, args.data_root)
    if args.ood_data_root is None:
        args.ood_data_root = args.data_root

    wandb_exp_name = (
        f"exp_id-{args.wandb_exp_id}_model-{args.model}_opt-{args.opt}_bs-{args.batch_size}_lr-{args.lr}"
    )
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        **build_wandb_init_kwargs(
            config=args,
            project=args.wandb_project_name,
            name=wandb_exp_name,
            entity=args.wandb_entity,
        )
    )

    print("\nWandb Setting:")
    print(f"\twandb_project_name: {args.wandb_project_name}")
    print(f"\twandb_exp_name: {wandb_exp_name}")

    exp_dict = wandb.config
    exp_dict["device"] = device

    print("\nExperimental Configuration:")
    for key, value in sorted(exp_dict.items()):
        print(f"\t{key}: {value}")

    ece_config = init_config()
    ece_config["num_reps"] = 100
    ece_config["norm"] = 1
    ece_config["ce_type"] = "ew_ece_bin"
    ece_config["num_bins"] = 10

    trained_model = train_loop(exp_dict, ece_config)

    if args.save_model:
        save_path = f"{args.save_model_path}/{args.model}_{args.opt}_{args.lr}.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
