# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:10:51 2019

@author: rehakomoon
"""

import torch
import torchvision
from pathlib import Path
from PIL import Image

class VRCDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, image_size):
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        
        self.file_list = [str(p.absolute()) for d in dataset_dir.iterdir() for p in d.glob('*.png')]
        self.num_data = len(self.file_list)
 
        self.transform_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        image = Image.open(self.file_list[idx])
        image = self.transform_flip(image)

        rotation_flag = torch.LongTensor(1).random_(0, 2)
        rotation_angle = torch.LongTensor(1).random_(0, 3) + 1
        label = (rotation_flag == 0).type(torch.LongTensor)
        image = torchvision.transforms.functional.rotate(image, label * (90 * rotation_angle))
        image = torchvision.transforms.ToTensor()(image)[0:3,:,:]

        return image, label

class VRCDatasetTest(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, image_size):
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        
        self.file_list = [str(p.absolute()) for d in dataset_dir.iterdir() for p in d.glob('*.png')]
        self.num_data = len(self.file_list)
        
    def __len__(self):
        return self.num_data * 8

    def __getitem__(self, idx):
        image_idx = idx//8
        flip_flag = (idx%8) < 4
        rotation_angle = idx%4
        label = torch.LongTensor([rotation_angle != 0])

        image = Image.open(self.file_list[image_idx])
        if flip_flag:
            image = torchvision.transforms.functional.hflip(image)
        
        image = torchvision.transforms.functional.rotate(image, 90 * rotation_angle)
        image = torchvision.transforms.ToTensor()(image)[0:3,:,:]

        return image, label
