# coding: utf-8
import math

import attr
import torch.optim as optim

from .adsgd import Adsgd
from .sam import SAM


OPTIMIZER_ALIASES = {
    "adsgd": "sasgd",
    "adsgd_sam": "sasam",
}

SUPPORTED_OPTIMIZERS = ("momentum_sgd", "adam", "sam", "sasgd", "sasam")
SMOOTHNESS_AWARE_OPTIMIZERS = frozenset({"sasgd", "sasam"})


def canonical_optimizer_name(name):
    return OPTIMIZER_ALIASES.get(name, name)


def requires_previous_model(name):
    return canonical_optimizer_name(name) in SMOOTHNESS_AWARE_OPTIMIZERS


@attr.s
class OptimizerSetting:
    name = attr.ib()
    lr = attr.ib()
    weight_decay = attr.ib()
    model = attr.ib()
    momentum = attr.ib(default=0.9)
    eps = attr.ib(default=1e-8)
    beta_1 = attr.ib(default=0.9)
    beta_2 = attr.ib(default=0.999)
    rho = attr.ib(default=0.05)
    amplifier = attr.ib(default=0.02)


def build_optimizer(setting: OptimizerSetting):
    name = canonical_optimizer_name(setting.name)
    model_params = setting.model.parameters()

    if name == "momentum_sgd":
        return optim.SGD(
            params=model_params,
            lr=setting.lr,
            momentum=setting.momentum,
            weight_decay=setting.weight_decay,
        )

    if name == "adam":
        return optim.Adam(
            params=model_params,
            lr=setting.lr,
            betas=(setting.beta_1, setting.beta_2),
            eps=setting.eps,
            weight_decay=setting.weight_decay,
            amsgrad=False,
        )

    if name == "sam":
        return SAM(
            params=model_params,
            base_optimizer=optim.SGD,
            rho=setting.rho,
            eps=setting.eps,
            adaptive_rho=False,
            lr=setting.lr,
            weight_decay=setting.weight_decay,
            momentum=setting.momentum,
        )

    if name == "sasam":
        return SAM(
            params=model_params,
            base_optimizer=Adsgd,
            rho=setting.rho,
            eps=setting.eps,
            adaptive_rho=True,
            lr=setting.lr,
            weight_decay=setting.weight_decay,
            amplifier=setting.amplifier,
        )

    if name == "sasgd":
        return Adsgd(
            params=model_params,
            lr=setting.lr,
            eps=setting.eps,
            weight_decay=setting.weight_decay,
            amplifier=setting.amplifier,
        )

    raise ValueError(
        f"Unsupported optimizer '{setting.name}'. Supported optimizers: {', '.join(SUPPORTED_OPTIMIZERS)}"
    )


def linear_warmup_cosine_decay(warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn
