# coding: utf-8
from functools import partial

import attr
from torch import Generator
from torch.utils.data import random_split
from torchvision import datasets, transforms


@attr.s
class DatasetSetting:
    name = attr.ib()
    root = attr.ib()
    split_ratio = attr.ib(default=0.8)
    split_seed = attr.ib(default=1)


@attr.s
class Dataset:
    name = attr.ib()
    num_classes = attr.ib()
    train_dataset = attr.ib()
    val_dataset = attr.ib()


def _transformer(train=True):
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose([transforms.ToTensor(), normalize])


def _split_train_val(train_val_dataset, split_ratio: float, split_seed: int):
    n_samples = len(train_val_dataset)
    train_size = int(n_samples * split_ratio)
    val_size = n_samples - train_size
    return random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=Generator().manual_seed(split_seed),
    )


def build_dataset(setting: DatasetSetting) -> Dataset:
    if setting.name != "cifar10":
        raise ValueError("Only 'cifar10' is supported in the paper reproduction code.")

    cifar10 = partial(datasets.CIFAR10, download=True)
    train_val_dataset = cifar10(setting.root, train=True, transform=_transformer(train=True))
    train_dataset, val_dataset = _split_train_val(
        train_val_dataset,
        setting.split_ratio,
        setting.split_seed,
    )

    print("\nDataset Summary:")
    print(f"\tlen(train_dataset): {len(train_dataset)}")
    print(f"\tlen(val_dataset): {len(val_dataset)}")

    return Dataset("cifar10", 10, train_dataset, val_dataset)
