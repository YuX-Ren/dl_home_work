import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from PIL import Image
Path = '.'

traindir = os.path.join(Path, 'train')
valdir = os.path.join(Path, 'val')
# valdir = os.path.join(args.data, 'val_reorg')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

print(train_dataset.class_to_idx)


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, main_dir, class_to_id, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.imgs_dir = os.path.join(main_dir, "images")
        self.all_imgs = os.listdir(self.imgs_dir)
        print(self.all_imgs)
        self.class_to_id = class_to_id

        self.annotations = {}
        with open(os.path.join(main_dir, "val_annotations.txt"), "r") as file:
            for line in file:
                items = line.split("\t")
                self.annotations[items[0]] = items[1]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.imgs_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        label = self.class_to_id[self.annotations[self.all_imgs[idx]]]  # Use the class to ID mapping

        if self.transform is not None:
            image = self.transform(image)

        return image, label


val_dataset = TinyImageNetDataset(
            valdir,
            train_dataset.class_to_idx,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))