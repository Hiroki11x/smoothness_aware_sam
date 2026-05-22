# coding: UTF-8

""" You can get cifar10.1 dataset from 
https://github.com/modestyachts/CIFAR-10.1/tree/master/datasets"""

import numpy as np
from PIL import Image
import os

ROOT_DIR=os.environ['DATA_DIR_PATH']

img_path = ROOT_DIR + "/CIFAR10_1/img_folder/"
label_path = ROOT_DIR + "/CIFAR10_1/CIFAR10_1.txt"


if __name__ == '__main__':
    
    cifar101_v4_data_path = ROOT_DIR + '/CIFAR10_1/cifar10.1_v4_data.npy'
    cifar101_v4_labels_path = ROOT_DIR + '/CIFAR10_1/cifar10.1_v4_labels.npy'
    cifar101_v4_data = np.load(cifar101_v4_data_path)
    cifar101_v4_labels = np.load(cifar101_v4_labels_path)

    num_sample = len(cifar101_v4_labels)

    #save image
    os.mkdir(img_path)
    for i in range(num_sample):
        np_img = cifar101_v4_data[i]
        np_img = Image.fromarray(np_img)
        np_img.save(img_path + str(i) + '.png')

    #save labels
    np_labels = cifar101_v4_labels
    with open(label_path, 'w') as writefile:
        for i in range(num_sample):
            writefile.write(img_path + str(i) + '.png' + ' ' +
                            str(int(np_labels[i])) + '\n')