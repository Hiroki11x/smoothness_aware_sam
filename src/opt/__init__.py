# coding: utf-8
import attr
import torch.optim as optim
from .sam import SAM
from .esam import ESAM
from .adsgd import Adsgd
import math

OPTIMIZER_ALIASES = {
    'adsgd': 'sasgd',
    'adsgd_sam': 'sasam',
}

SMOOTHNESS_AWARE_OPTIMIZERS = frozenset({'sasgd', 'sasam'})


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

    momentum = attr.ib(default=0.9) # sgd, sgd_nesterov
    eps = attr.ib(default=1e-8) # adam, rmsprop (term added to the denominator to improve numerical stability )

    beta_1 = attr.ib(default=0.9) #adam
    beta_2 = attr.ib(default=0.999) #adam

    rho = attr.ib(default=0.05) #rho
    esam_beta = attr.ib(default=0.5) #esam
    esam_gamma = attr.ib(default=1.0) #esam

    amplifier = attr.ib(default=0.02) #adsgd

def build_optimizer(setting: OptimizerSetting):
    name = canonical_optimizer_name(setting.name)
    model_params = setting.model.parameters()

    # Standard Optimizer
    if name == 'vanilla_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        weight_decay=setting.weight_decay)

    elif name == 'momentum_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        momentum=setting.momentum,
                        weight_decay=setting.weight_decay)

    elif name == 'nesterov_momentum_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        momentum=setting.momentum, 
                        weight_decay=setting.weight_decay, 
                        nesterov=True)

    elif name == 'adam':
        return optim.Adam(params=model_params, 
                        lr=setting.lr, 
                        betas=(setting.beta_1, setting.beta_2), 
                        eps=setting.eps, 
                        weight_decay=setting.weight_decay, 
                        amsgrad=False)

    elif name == 'adamw':
        return optim.AdamW(params=model_params, 
                        lr=setting.lr, 
                        betas=(setting.beta_1, setting.beta_2), 
                        eps=setting.eps, 
                        weight_decay=setting.weight_decay, 
                        amsgrad=False)

    elif name == 'sam':
        return SAM(params=model_params, 
                   base_optimizer=optim.SGD,
                   rho=setting.rho,
                   eps=setting.eps,
                   adaptive_rho=False,
                   lr=setting.lr, 
                   weight_decay=setting.weight_decay, 
                   momentum=setting.momentum)

    elif name == 'esam':
        return ESAM(params=model_params, 
                    base_optimizer=optim.SGD,
                    rho=setting.rho,
                    beta=setting.esam_beta,
                    gamma=setting.esam_gamma,
                    eps=setting.eps,
                    lr=setting.lr, 
                    weight_decay=setting.weight_decay, 
                    momentum=setting.momentum)
    
    elif name == 'sasam':
        return SAM(params=model_params,
                    base_optimizer=Adsgd,
                    rho=setting.rho,
                    eps=setting.eps,
                    adaptive_rho=True,
                    lr=setting.lr,
                    weight_decay=setting.weight_decay, 
                    amplifier=setting.amplifier)
    
    elif name == 'sasgd':
        return Adsgd(params=model_params,
                     lr=setting.lr,
                     eps=setting.eps, 
                     weight_decay=setting.weight_decay, 
                     amplifier=setting.amplifier)
    

    else:
        raise ValueError(
            'The selected optimizer is not supported for this trainer.')

      
def linear_warmup_cosine_decay(warmup_steps, total_steps):

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn
