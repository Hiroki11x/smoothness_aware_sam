# ref https://github.com/RLOptim/OOD_IMGEXP/blob/2d366b84d133723fc9e2fe88308b788c11002b61/src/util/exp_util.py

import torch
import os
from .data_list import ImageList
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .preprocess.cifar10_c import CIFAR10C

#--------------------------------------- eval dataset ----------------------------------------
def eval_loader(exp_dict, kind):

    dataset_name = exp_dict['ood_dataset']
    abs_path = os.environ['DATA_DIR_PATH']

    cifar10_dataset = ['cifar10_1', 'cifar10_c', 'cifar10_p']
    imagenet_dataset = ['imagenet_v2', 'imagenet_sketch', 'imaganet_c']
    
    if dataset_name == 'cifar10_1':
            
        labels_path = abs_path + '/CIFAR10_1/CIFAR10_1.txt'
        img_size = 32

        labels = open(labels_path).readlines()
        loader = torch.utils.data.DataLoader(
            ImageList(labels, transform=transforms.Compose([
                # transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]), mode='RGB'),
            batch_size=exp_dict['batch_size'], shuffle=False, num_workers=exp_dict['num_workers'], pin_memory=True)

        return loader


    elif dataset_name == 'cifar10_c':

        print(f'cifar10_c is called / corruption_kind: {kind}')

        corruption_kind = kind
        root_path = abs_path + '/CIFAR10_C/npy_files'
        # loaded_data = load_cifar10_c(root=root_path, corruption_kind=corruption_kind)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], 
                                 [0.24703223, 0.24348513, 0.26158784])
        ])

        dataset = CIFAR10C(
                    root_path,
                    corruption_kind, transform=transform
        )

        loader = DataLoader(dataset, batch_size=exp_dict['batch_size'], shuffle=False, num_workers=4)

        return loader


    elif dataset_name in imagenet_dataset:
        if dataset_name == 'imagenet_v2':
            print(f'imagenet_v2 is called / variant: {kind}')

            labels_path = abs_path + f"/ImagenNet-V2/ImageNetV2-{kind}.txt"

            labels = open(labels_path).readlines()
            loader = torch.utils.data.DataLoader(
                ImageList(labels, transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]), mode='RGB'),
                batch_size=exp_dict['batch_size'], shuffle=False, num_workers=exp_dict['num_workers'], pin_memory=True)
            return loader
        
    else:
        print(f'[build eval_loader] this dataset is not supported yet. / dataset_name: {dataset_name}')
        return None
