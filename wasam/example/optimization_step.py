import torch
from sam import SAM, disable_running_stats, enable_running_stats
# from wasam import WASAM, disable_running_stats, enable_running_stats

from example.config import TrainConfig


# def get_gradient_norm(model: torch.nn.Module) -> float:
#     """Computes gradient norm of model."""
#     total_norm = 0.0
#     for p in model.parameters():
#         param_norm = p.grad.data.norm(2)
#         total_norm += param_norm.item() ** 2
#     total_norm = total_norm**0.5
#     return total_norm

def get_gradient_norm(model: torch.nn.Module) -> float:
    """Computes gradient norm of model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None: # for EfficientNet
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


# SAM
# https://github.com/JeanKaddour/WASAM
def SAM_optimization_step(
    model: torch.nn.Module,
    optimizer: SAM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    grad_clip: float,
) -> dict:
    # first forward-backward pass (use this pass for logging)
    enable_running_stats(model)
    predictions = model.forward(inputs)
    loss = criterion(input=predictions, target=labels)
    loss.backward()

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.first_step()

    # second forward-backward pass (ignore this pass' statistics)
    disable_running_stats(model)
    criterion(input=model.forward(inputs), target=labels).backward()
    gradient_norm = get_gradient_norm(model=model)
    optimizer.second_step()
    step = {
        "loss": loss.item(),
        "grad_norm": gradient_norm,
        "predictions": predictions,
    }
    return step

# SA-SAM
# https://github.com/Hiroki11x/SAM_Calibration/blob/08bb2295283c6bb5f6c8562d5acd422d1e86e718/src/algorithm/algo_erm.py#L116
def SASAM_optimization_step(
    model: torch.nn.Module,
    prev_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    prev_optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    grad_clip: float,
) -> dict:
    # first forward-backward pass (use this pass for logging)
    enable_running_stats(model)
    predictions = model.forward(inputs)
    loss = criterion(input=predictions, target=labels)
    loss.backward()

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Smoothness Aware
    prev_prediction=prev_model.forward(inputs)
    prev_loss=criterion(input=prev_prediction, target=labels)
    prev_loss.backward()
    optimizer.base_optimizer.compute_dif_norms(prev_optimizer) 
    prev_model.load_state_dict(model.state_dict())

    optimizer.first_step()

    # second forward-backward pass (ignore this pass' statistics)
    disable_running_stats(model)
    criterion(input=model.forward(inputs), target=labels).backward()
    gradient_norm = get_gradient_norm(model=model)
    optimizer.second_step()
    step = {
        "loss": loss.item(),
        "grad_norm": gradient_norm,
        "predictions": predictions,
    }
    prev_optimizer.zero_grad(set_to_none=True)
    return step

# SA-SGD
# https://github.com/Hiroki11x/SAM_Calibration/blob/08bb2295283c6bb5f6c8562d5acd422d1e86e718/src/algorithm/algo_erm.py#L104
def SASGD_optimization_step(
    model: torch.nn.Module,
    prev_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    prev_optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    grad_clip: float,
) -> dict:
    predictions = model.forward(inputs)
    loss = criterion(input=predictions, target=labels)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    gradient_norm = get_gradient_norm(model=model)

    prev_prediction=prev_model.forward(inputs)
    prev_loss=criterion(input=prev_prediction, target=labels)
    prev_loss.backward()

    optimizer.compute_dif_norms(prev_optimizer)
    prev_model.load_state_dict(model.state_dict())

    optimizer.step()

    step = {
        "loss": loss.item(),
        "grad_norm": gradient_norm,
        "predictions": predictions,
    }
    optimizer.zero_grad(set_to_none=True)
    prev_optimizer.zero_grad(set_to_none=True)
    return step

# SGD, Adam
# https://github.com/JeanKaddour/WASAM
def one_optimization_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    grad_clip: float,
) -> dict:

    predictions = model.forward(inputs)
    loss = criterion(input=predictions, target=labels)
    loss.backward()

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    gradient_norm = get_gradient_norm(model=model)
    optimizer.step()
    step = {
        "loss": loss.item(),
        "grad_norm": gradient_norm,
        "predictions": predictions,
    }
    optimizer.zero_grad()
    return step

# Main Function
def optimization_step(
    model: torch.nn.Module,
    prev_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    prev_optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    config: TrainConfig,
) -> dict:
    # SAM
    if config.optimizer_name == "sam":
        return SAM_optimization_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            labels=labels,
            criterion=criterion,
            grad_clip=config.grad_clip,
        )
    # Smoothness Aware SGD
    elif config.optimizer_name == "sasgd":
        return SASGD_optimization_step(
            model=model,
            prev_model=prev_model,
            optimizer=optimizer,
            prev_optimizer=prev_optimizer,
            inputs=inputs,
            labels=labels,
            criterion=criterion,
            grad_clip=config.grad_clip,
        )
    # Smoothness Aware SAM
    elif config.optimizer_name == "sasam":
        return SASAM_optimization_step(
            model=model,
            prev_model=prev_model,
            optimizer=optimizer,
            prev_optimizer=prev_optimizer,
            inputs=inputs,
            labels=labels,
            criterion=criterion,
            grad_clip=config.grad_clip,
        )
    # SGD, Adam
    else:
        return one_optimization_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            labels=labels,
            criterion=criterion,
            grad_clip=config.grad_clip,
        )
    