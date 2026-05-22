# coding: utf-8
import attr
import torch.nn
import timm
from torchvision import models
from .vit_small import ViT_Small
from .vit import ViT
from .resnet_cifar import ResNet18_1_Cifar, ResNet18_2_Cifar, ResNet18_3_Cifar
from .resnet_imagenet import ResNet18_1, ResNet18_2, ResNet18_3, ResNet50_1, ResNet50_2, ResNet152_05, ResNet152_1
from .mlp import Medium_MLP

from .resnet_mnist import ResNet18_1_Mnist
from .mlp_mnist import MLP_Mnist
# from .simple_convnext import SimpleConvNext
from .convnext import ConvNeXt
from .mobilenet_v2 import MobileNetV2
from .efficientnet import EfficientNetB0
from .vgg import VGG

@attr.s
class ModelSetting:
    name = attr.ib()
    num_classes = attr.ib()
    num_dim = attr.ib(4096)
    # dropout_ratio = attr.ib()


def build_model(setting: ModelSetting):
    name = setting.name
    num_classes = setting.num_classes
    num_dim = setting.num_dim
    # dropout_ratio = setting.dropout_ratio


    # ============================================== MNIST ==============================================
    # ============================================== MNIST ==============================================
    # ============================================== MNIST ==============================================
    # ============================================== MNIST ==============================================
    # ============================================== MNIST ==============================================
    
    if name == 'resnet18_1_mnist':
        model = ResNet18_1_Mnist(num_classes)
        return model

    if name == 'mlp_mnist':
        model = MLP_Mnist(num_classes)
        return model
    

    # ============================================== cifar10 ==============================================
    # ============================================== cifar10 ==============================================
    # ============================================== cifar10 ==============================================
    # ============================================== cifar10 ==============================================
    # ============================================== cifar10 ==============================================


    # [EfficientNet]
    if name == 'efficientnet_b0':
        model = EfficientNetB0()
        return model



    # ============================================== cifar10 or cifar100 ==============================================
    # ============================================== cifar10 or cifar100 ==============================================
    # ============================================== cifar10 or cifar100 ==============================================
    # ============================================== cifar10 or cifar100 ==============================================
    # ============================================== cifar10 or cifar100 ==============================================

    # [VGG]
    if name == 'vgg19':
        model = VGG('VGG19', num_classes=num_classes)
        return model

    # [MobileNet]
    if name == 'mobilenet_v2':
        model = MobileNetV2(num_classes=num_classes)
        return model

    # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

    # [ConvNext]
    # https://github.com/filipbasara0/simple-convnext/blob/60b9f8a6bb9e28a2feed49f3012e068555fe17a8/train.py#L40
    # if name == 'simple_convnext':
    #     model = SimpleConvNext(num_channels=3,
    #                 num_classes=num_classes,
    #                 patch_size=4,
    #                 layer_dims=[64, 128, 256, 512],
    #                 depths=[2, 2, 2, 2],
    #                 drop_rate=0.0)
    #     return model


    # https://juliusruseckas.github.io/ml/convnext-cifar10.html
    if name == 'convnext':
        model = ConvNeXt(num_classes,
                 channel_list = [64, 128, 256, 512],
                 num_blocks_list = [2, 2, 2, 2],
                 kernel_size=7, patch_size=1,
                 res_p_drop=0.)
        return model

    # ref https://github.com/insilicomab/pytorch-intro-hydra-wandb/blob/2615433b67cfb7310b6f83c8210e63b3c9fe18f4/notebooks/training.ipynb
    # if name == 'convnext_small':
    #     model = timm.create_model(
    #         'convnext_small', 
    #         pretrained=False, 
    #         num_classes=num_classes
    #     )
    #     return model


    # [Transformer]
    # ref https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/models/vit_small.py
    # ref https://github.com/lucidrains/vit-pytorch#vision-transformer-for-small-datasets
    if name == 'vit_small':
        model = ViT_Small(
            image_size = 32, 
            patch_size = 4,
            num_classes = num_classes,
            dim = 512, 
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0,
            emb_dropout = 0
        )
        return model


    # ref https://github.com/lucidrains/vit-pytorch#faq
    if name == 'vit':
        model = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0,
            emb_dropout = 0
        )
        return model


    # [MLP]
    if name == 'medium_mlp':
        print('num_dim', num_dim)
        model = Medium_MLP(num_classes, num_dim=num_dim)
        return model

    # [ResNet]
    if name == 'resnet18_1_cifar':
        model = ResNet18_1_Cifar(num_classes)
        return model

    if name == 'resnet18_2_cifar':
        model = ResNet18_2_Cifar(num_classes)
        return model

    if name == 'resnet18_3_cifar':
        model = ResNet18_3_Cifar(num_classes)
        return model

    # ============================================== ImageNet ==============================================
    # ============================================== ImageNet ==============================================
    # ============================================== ImageNet ==============================================
    # ============================================== ImageNet ==============================================
    # ============================================== ImageNet ==============================================
    if name == 'resnet18_1':
        model = ResNet18_1(num_classes)
        return model

    if name == 'resnet18_2':
        model = ResNet18_2(num_classes)
        return model

    if name == 'resnet18_3':
        model = ResNet18_3(num_classes)
        return model

    if name == 'resnet50_1':
        model = ResNet50_1(num_classes)
        return model

    if name == 'resnet50_2':
        model = ResNet50_2(num_classes)
        return model

    if name == 'resnet152_05':
        model = ResNet152_05(num_classes)
        return model

    if name == 'resnet152_1':
        model = ResNet152_1(num_classes)
        return model


    raise ValueError('The selected model is not supported for this trainer.')