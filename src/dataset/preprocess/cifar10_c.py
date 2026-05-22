import os

import numpy as np
from PIL import Image
from torchvision import datasets


corruptions = [
    "fog",
    "jpeg_compression",
    "zoom_blur",
    "speckle_noise",
    "glass_blur",
    "spatter",
    "shot_noise",
    "defocus_blur",
    "elastic_transform",
    "gaussian_blur",
    "frost",
    "saturate",
    "brightness",
    "snow",
    "gaussian_noise",
    "motion_blur",
    "contrast",
    "impulse_noise",
    "pixelate",
]


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root: str, name: str, transform=None, target_transform=None):
        if name not in corruptions:
            raise ValueError(f"Unsupported corruption '{name}'.")
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = np.load(os.path.join(root, f"{name}.npy"))
        self.targets = np.load(os.path.join(root, "labels.npy"))

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.data)
