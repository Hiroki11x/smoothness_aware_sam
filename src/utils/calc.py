import torch
import numpy as np

def get_grads(model):
    params = list(model.parameters())
    grads = []
    for param_group in params:
        grads.append(param_group.grad.flatten())
    grads = torch.cat(grads)
    return grads

def get_weights(model):
    params = list(model.parameters())
    weights = []
    for param_group in params:
        weights.append(param_group.flatten())
    weights = torch.cat(weights)
    return weights


