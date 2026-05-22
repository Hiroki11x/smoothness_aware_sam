# coding; utf-8
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
    # For ImageNet
    # root_val = attr.ib(default=None)


@attr.s
class Dataset:
    name = attr.ib()
    num_classes = attr.ib()
    train_dataset = attr.ib()
    val_dataset = attr.ib()
    test_dataset = attr.ib(default=None)


def _transformer(data_name, train=True):

    if data_name == 'imagenet':
        raise NotImplementedError

    elif data_name == 'cifar10' or data_name == 'cifar100':
        if train:

            # ref https://github.com/torrvision/focal_calibration/blob/main/Data/cifar100.py
            # ref https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/main.py#L31
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
    elif data_name == 'mnist':
        if train:
            return transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) 
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) 
            ])

    else:
        raise NotImplementedError



def _split_train_val(train_val_dataset, split_ratio:float, split_seed:int):
    n_samples = len(train_val_dataset)
    train_size = int(n_samples * split_ratio)
    val_size = n_samples - train_size
    return random_split(
        train_val_dataset, [train_size, val_size], generator=Generator().manual_seed(split_seed)
    )


def build_dataset(setting: DatasetSetting) -> Dataset:
    name = setting.name
    root = setting.root
    split_ratio = setting.split_ratio
    split_seed = setting.split_seed
    datasets_f = {
        'cifar10': partial(datasets.CIFAR10, download=True),
        'cifar100': partial(datasets.CIFAR100, download=True),
        'mnist': partial(datasets.MNIST, download=True),
        'imagenet': None
    }[name]

    if name == 'imagenet':
        train_root = "{}/train".format(root)
        val_root = "{}/val".format(root)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        # refer https://github.com/jiweibo/ImageNet/tree/master (ResNet50 76.130% Top1 Acc)
        # https://github.com/jiweibo/ImageNet/blob/master/data_loader.py
        
        train_dataset = \
            datasets.ImageFolder(root=train_root,
                                transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize
                            ])
            )
            
        # refer https://github.com/jiweibo/ImageNet/tree/master (ResNet50 76.130% Top1 Acc)
        # https://github.com/jiweibo/ImageNet/blob/master/data_loader.py
        val_dataset = \
            datasets.ImageFolder(root=val_root,
                                transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize
                            ])
            )


        print("\nDataset Summary:")
        print(f'\tlen(train_dataset): {len(train_dataset)}')
        print(f'\tlen(val_dataset): {len(val_dataset)}')
        # print(f'\tlen(test_dataset): {len(test_dataset)}')
        # return Dataset(name, 1000, train_dataset, val_dataset, test_dataset)

        return Dataset(name, 1000, train_dataset, val_dataset)
    
    elif name == 'cifar10':        

        train_val_dataset = datasets_f(
            root, train=True, transform=_transformer(name, True))
        train_dataset, val_dataset = _split_train_val(train_val_dataset, split_ratio, split_seed)
        test_dataset = datasets_f(
            root, train=False, transform=_transformer(name, False))

        print("\nDataset Summary:")
        print(f'\tlen(train_dataset): {len(train_dataset)}')
        print(f'\tlen(val_dataset): {len(val_dataset)}')
        
        return Dataset(name, 10, train_dataset, val_dataset)

    elif name == 'cifar100':

        train_val_dataset = datasets_f(
            root, train=True, transform=_transformer(name, True))
        train_dataset, val_dataset = _split_train_val(train_val_dataset, split_ratio, split_seed)
        test_dataset = datasets_f(
            root, train=False, transform=_transformer(name, False))

        print("\nDataset Summary:")
        print(f'\tlen(train_dataset): {len(train_dataset)}')
        print(f'\tlen(val_dataset): {len(val_dataset)}')

        return Dataset(name, 100, train_dataset, val_dataset)

    elif name == 'mnist':

        train_val_dataset = datasets_f(
            root, train=True, transform=_transformer(name, True))
        train_dataset, val_dataset = _split_train_val(train_val_dataset, split_ratio, split_seed)
        test_dataset = datasets_f(
            root, train=False, transform=_transformer(name, False))

        print("\nDataset Summary:")
        print(f'\tlen(train_dataset): {len(train_dataset)}')
        print(f'\tlen(val_dataset): {len(val_dataset)}')

        return Dataset(name, 10, train_dataset, val_dataset)

    else:
        raise NotImplementedError
    
