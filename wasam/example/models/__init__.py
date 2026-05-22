import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models

# from math import round
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from example.models.vit import VisionTransformer
from example.models.pyramidnet import PyramidNet
from example.models.wrn import WideResNet
from example.models.vgg import VGG16
from example.models.vit_small import ViT_Small
from example.models.mobilenet_v2 import MobileNetV2

MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    # "wide_resnet50_2": models.wide_resnet50_2,
    # "wide_resnet101_2": models.wide_resnet101_2,
    # "resnext101_32x8d": models.resnext101_32x8d,
    # "densenet121": models.densenet121,
    # "efficientnet_b7": models.efficientnet_b7,
    # "efficientnet_b5": models.efficientnet_b5,
    "pyramid": PyramidNet,
    "wrn": WideResNet,
    "vit": VisionTransformer,
    "vgg": VGG16,
    "vit_small": ViT_Small,
    "mobilenet_v2": MobileNetV2,
}
