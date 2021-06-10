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

class MiniImagenet(BaseDataset):
    def __init__(self, cfg, phase="train"):
        super().__init__(cfg, phase)
        self.transform = self.prepare_transform(cfg, phase)

    def prepare_transform(self, cfg, phase):
        norm = transforms.Normalize(
            np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
            np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
        )
        if cfg.model.encoder == "FourLayer_64F":
            if phase == "train":
                t = [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize([84, 84]),
                    transforms.ToTensor(),
                    norm
                ]
            else:
                t = [
                    transforms.Resize([84, 84]),
                    transforms.ToTensor(),
                    norm
                ]
        else:
            if phase == "train":
                t = [
                    transforms.Resize([92, 92]),
                    transforms.RandomCrop(84),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    norm
                ]
            else:
                t = [
                    transforms.Resize([92, 92]),
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    norm
                ]
        return transforms.Compose(t)
