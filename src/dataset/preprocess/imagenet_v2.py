from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
import torch

import shutil
import os
import glob


ROOT_DIR=os.environ['DATA_DIR_PATH']

# Download Dataset and Untar

'''
https://github.com/modestyachts/ImageNetV2_pytorch

pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

Rio: DONE
Mila: DONE
Tokyo Tech: DONE
'''


variants = ["matched-frequency", "threshold-0.7", "top-images"]
# variants = ["matched-frequency", "threshold-0.7"]


if __name__ == '__main__':
    
    for varinat in variants:

        dataset_path = ROOT_DIR + "/ImagenNet-V2"
        dataset = ImageNetV2Dataset(varinat, location=dataset_path) 

        # Create Label.txt
        orig_file = ROOT_DIR + f"/ImagenNet-V2/ImageNetV2-{varinat}/"
        files = os.listdir(orig_file)
        len_files = len(files)
        txt_name = ROOT_DIR + f'/ImagenNet-V2/ImageNetV2-{varinat}.txt'

        with open(txt_name, 'w') as writefile:
            for k in range(len_files):
                x = orig_file + files[k]
                a = glob.glob(orig_file + files[k] + '/*')
                for l in range(len(a)):
                    writefile.write(a[l] + ' ' + str(files[k]) + '\n')
