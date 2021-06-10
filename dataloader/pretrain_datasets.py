import os
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image

class PreDataset(data.Dataset):
    def __init__(self, cfg, phase="train"):
        super().__init__()

        self.data_list = self.prepare_data_list(cfg, phase)
        self.transform = self.prepare_transform(cfg, phase)

        self.label = [l for (d, l) in self.data_list]

    def prepare_transform(self, cfg, phase):
        x = 1
        data_folder = osp.basename(osp.abspath(cfg.data.image_dir))

        if data_folder == "Cifar-100":
            norm = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
        else:
            norm = transforms.Normalize(
                np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
            )

        if phase == "train":
            t = [
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomResizedCrop(84),
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

    def prepare_data_list(self, cfg, phase):
        folder = osp.join(cfg.data.image_dir, phase)
        
        class_folders = [osp.join(folder, label) \
            for label in os.listdir(folder) \
            if osp.isdir(osp.join(folder, label)) \
        ]
        random.shuffle(class_folders)

        x_list = []
        y_list = []
        for i, cls in enumerate(class_folders):
            imgs = [osp.join(cls, img) for img in os.listdir(cls) if ".png" in img or ".jpg" in img]
            x_list = x_list + imgs
            y_list = y_list + [i] * len(imgs)
        data_list = list(zip(x_list, y_list))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        x, y = self.data_list[index]
        im_x = self.transform(Image.open(x).convert("RGB"))
        return im_x, y

