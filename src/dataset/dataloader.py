import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .preprocess.cifar10_c import CIFAR10C


def _resolve_cifar10_c_root(exp_dict):
    base_root = exp_dict.get("ood_data_root") or exp_dict["data_root"]
    candidates = (
        base_root,
        os.path.join(base_root, "CIFAR10_C"),
        os.path.join(base_root, "CIFAR10_C", "npy_files"),
    )
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "labels.npy")):
            return candidate
    return os.path.join(base_root, "CIFAR10_C", "npy_files")


def eval_loader(exp_dict, corruption_kind):
    if exp_dict["ood_dataset"] != "cifar10_c":
        raise ValueError("Only 'cifar10_c' is supported in the paper reproduction code.")

    root_path = _resolve_cifar10_c_root(exp_dict)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.49139968, 0.48215841, 0.44653091],
                [0.24703223, 0.24348513, 0.26158784],
            ),
        ]
    )

    dataset = CIFAR10C(root_path, corruption_kind, transform=transform)
    return DataLoader(
        dataset,
        batch_size=exp_dict["batch_size"],
        shuffle=False,
        num_workers=exp_dict["num_workers"],
    )
