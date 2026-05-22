'''
https://github.com/torrvision/focal_calibration/blob/main/Losses/loss.py
Implementation of the following loss functions:
1. Cross Entropy
2. Focal Loss
'''

from torch.nn import functional as F
from .focal_loss import FocalLoss


def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs['gamma'])(logits, targets)