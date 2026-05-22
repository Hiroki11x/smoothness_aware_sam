import random
from datetime import datetime
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import sys

# import types
# sys.modules['psutil'] = types.ModuleType('psutil')

import wandb
from torch import nn as nn
from torch.optim.swa_utils import AveragedModel

from example.config import TrainConfig
from example.label_smoothing import LabelSmoothingLoss

# from example.models import MODELS, PyramidNet, WideResNet
# from wasam import WASAM

from sam import SAM
from adsgd import Adsgd

import os
import torchvision
import PIL

# from example.models.model_config import get_b16_config
# from example.models.vit import VisionTransformer
from example.models.vit_small import ViT_Small
from example.models.vgg import VGG19
from example.models.wrn import WideResNet
from example.models.pyramidnet import PyramidNet
from example.models.mobilenet_v2 import MobileNetV2

import threading

def handle_threading_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, PermissionError):
        print(f"PermissionError caught: {exc_value}")
        return
    threading.excepthook(exc_type, exc_value, exc_traceback)

def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_starting_time() -> str:
    return "{:%Y_%m_%d_%H_%M_%S_%f}".format(datetime.now())


def build_wandb_init_kwargs(config: TrainConfig, project: str, name: str) -> dict:
    kwargs = {
        "project": project,
        "name": name,
        "config": config,
    }
    if config.wandb_entity:
        kwargs["entity"] = config.wandb_entity
    return kwargs

def print_libs_version():
    print("\nComputational Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Fast Mode
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Repro Mode
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def init(config: TrainConfig) -> None:

    print_libs_version()

    project_name = config.wandb_project_name

    specific_setting = f'{config.optimizer_name}'
    if config.is_swa:
        specific_setting = f'{specific_setting}_swa'
    if config.is_sa:
        specific_setting = f'{specific_setting}_sa'
    if config.sam_rho_fixed:
        specific_setting = f'{specific_setting}_rho_fixed'
        
    run_name = f"{config.wandb_exp_id}-{specific_setting}-{config.dataset_name}-{config.model}-{config.max_epochs}-{str(config.seed)}"

    if config.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"
        print('wandb init as dryrun')
        wandb.init(**build_wandb_init_kwargs(
            config=config,
            project=project_name,
            name=run_name,
        ))
    else:
        print('wandb init as online')
        wandb.init(**build_wandb_init_kwargs(
            config=config,
            project=project_name,
            name=run_name,
        ))
        
    set_seeds(config.seed)
    print('\nExperimental Configuration:')
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
        for k, v in sorted(config_dict.items()):
            print('\t{}: {}'.format(k, v))
    else:
        print('No configuration settings found.')


from typing import Dict

def log_results(accuracies: Dict[str, float], global_step: int, suffix: str = ""):
    wandb.log(
        {
            "global_step": global_step,
            f"valid{suffix}": accuracies["valid"],
            f"test{suffix}": accuracies["test"],
        }
    )
    if accuracies["valid"] > wandb.run.summary[f"valid.best_valid{suffix}"]:
        wandb.run.summary[f"best_valid{suffix}"] = accuracies["valid"]
        wandb.run.summary[f"best_test{suffix}"] = accuracies["test"]
        wandb.run.summary[f"best_step{suffix}"] = global_step


def init_wandb_results(swa_start_times: List[int]) -> None:
    metrics = [
        f"valid",
        f"test",
    ]
    strings = metrics[:]
    if swa_start_times is not None:
        for start_time in swa_start_times:
            for metric in metrics:
                strings.append(metric + f"_swa_{start_time}")
    for string in strings:
        wandb.run.summary[f"{string}_best_acc"] = 0.0


def get_device(config: TrainConfig) -> Optional[torch.device]:
    return (
        torch.device(f"cuda:{config.device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

from typing import List

class NoneScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def step(self) -> None:
        pass

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, scheduler, warmup_steps, initial_lr, last_epoch=-1):
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.final_lr = initial_lr  # 最終的にwarmup後の学習率をこの値に設定
        self.step_count = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr_scale = (self.final_lr / self.initial_lr) * (self.last_epoch / self.warmup_steps)
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            return self.scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch <= self.warmup_steps:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            self.scheduler.step(epoch - self.warmup_steps)  # warmupが完了したら、元のスケジューラーを使用



def get_lr_scheduler(
    config: TrainConfig, optimizer: torch.optim.Optimizer
) -> Union[
    torch.optim.lr_scheduler.ExponentialLR,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    torch.optim.lr_scheduler.CyclicLR,
    torch.optim.lr_scheduler.StepLR,
    torch.optim.lr_scheduler.MultiStepLR,
    NoneScheduler,
]:
    # See https://github.com/davda54/sam/issues/28
    # if "sam" in config.optimizer_name: # Modify
    #     optimizer = optimizer.base_optimizer
    if config.total_steps <= 0:
        raise ValueError("total_steps must be greater than 0")

    if config.lr_scheduler == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.total_steps)
    elif config.lr_scheduler == "none":
        base_scheduler = NoneScheduler(optimizer)
    else:
        raise NotImplementedError(f"Scheduler {config.lr_scheduler} not implemented")

    
    if config.is_warmup:
        # WarmupScheduler
        warmup_steps = 500
        initial_lr = 1e-5
        return WarmupScheduler(optimizer, base_scheduler, warmup_steps, initial_lr)    
    else:
        return base_scheduler



def get_criterion(config: TrainConfig) -> torch.nn.Module:
    if config.loss == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif config.loss == "label_smoothing":
        return LabelSmoothingLoss(config.label_smoothing_factor)
    raise NotImplementedError(f"Loss {config.loss} not implemented")


def get_SWA_model(
    config: TrainConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> Optional[AveragedModel]:
    swa_model, swa_lr = None, None
    if config.is_swa:
        swa_model = AveragedModel(model=model, device=device)
    return swa_model


def get_optimizer(
    optimizer_name: str, config: TrainConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    if optimizer_name == "sam":
        return SAM(
            params=model.parameters(),
            base_optimizer=get_optimizer(
                optimizer_name="sgd", config=config, model=model
            ),
            rho=config.sam_rho,
            rho_fixed=config.sam_rho_fixed,
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "sasam":
        return SAM(
            params=model.parameters(),
            base_optimizer=get_optimizer(
                optimizer_name="sasgd", config=config, model=model
            ),
            rho=config.sam_rho,
            rho_fixed=config.sam_rho_fixed,
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            adaptive=True,
        )
    elif optimizer_name == "sasgd":
        return Adsgd(
            params=model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            amplifier=config.amplifier,
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            params=model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta_1, config.adam_beta_2),
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta_1, config.adam_beta_2),
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            nesterov=config.sgd_nesterov,
        )
    raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")


def get_model(config: TrainConfig) -> nn.Module:
    # assert config.model in MODELS, f"Invalid model name: {config.model}."

    if config.model == "pyramid":
        model = PyramidNet(
            dataset=config.dataset_name,
            depth=110,
            alpha=270,
            num_classes=config.num_classes,
        )
    elif config.model == "wrn":
        model = WideResNet(
            depth=28,
            width_factor=10,
            dropout=config.dropout,
            in_channels=3,
            labels=config.num_classes,
        )

    # elif config.model == "vit":
    #     # https://github.com/kentaroy47/vision-transformers-cifar10/blob/03690f363c2be92b5c0568b8d0f0ef494266a91d/train_cifar10.py#L173

    #     vit_config = get_b16_config()
    #     model = VisionTransformer(vit_config, config.img_size, zero_head=True, num_classes=config.num_classes)
    #     model.load_from(np.load(config.pretrained_path))
    # additional model archs other than the original paper (https://arxiv.org/pdf/2202.00661)

    elif config.model == 'vgg':
        model = VGG19(num_classes=config.num_classes)

    elif config.model == 'vit_small':
        model = ViT_Small(
            image_size = 32, 
            patch_size = 4,
            num_classes = config.num_classes,
            dim = 512, 
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0,
            emb_dropout = 0
        )

    elif config.model == 'mobilenet_v2':
        model = MobileNetV2(num_classes=config.num_classes)

    else:
        raise NotImplementedError(f"Model {config.model} not supported.")
        # model = MODELS[config.model](
        #     pretrained=config.pretrained, num_classes=config.num_classes
        # )

    return model
