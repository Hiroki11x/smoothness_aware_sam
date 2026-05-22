import torch
from typing import Any, Callable, Dict, Iterable, Optional, Union
from torch.nn.modules.batchnorm import _BatchNorm

Params = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
State = Dict[str, Any]
LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]

class SAM(torch.optim.Optimizer):
    def __init__(
        self,
        params: Params,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        rho_fixed: bool = False, 
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        # defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        defaults = dict(rho=rho, adaptive=adaptive, rho_fixed=rho_fixed, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        grad_norm = self._grad_norm()

        # for group in self.param_groups: # Modify
        for group, base_group in zip(self.param_groups, self.base_optimizer.param_groups):
            if not group['rho_fixed']: #For SASGD 
                group['rho'] = base_group['lr'] ** 0.5 
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    def step(self, closure: OptLossClosure = None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)