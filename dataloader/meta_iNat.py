import os
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
import csv

from .base_datasets import BaseDataset

class meta_iNat(BaseDataset):
    def __init__(self, cfg, phase="train"):
        super().__init__(cfg, phase)
        self.transform = self.prepare_transform(cfg, phase)

    def prepare_transform(self, cfg, phase):
        norm = transforms.Normalize(
            np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
            np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
        )
        if phase == "train":
            t = [
                transforms.RandomCrop(84,padding=8,padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm
            ]
        else:
            t = [
                transforms.ToTensor(),
                norm
            ]
        return transforms.Compose(t)
