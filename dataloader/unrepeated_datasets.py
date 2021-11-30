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
from copy import deepcopy

class UnrepeatedDataset(BaseDataset):
    def __init__(self, cfg, phase="train", transform=None):
        super().__init__(cfg, phase, transform)

        self.data_list = self.prepare_data_list(cfg, phase)

    def prepare_data_list(self, cfg, phase):
        folder = osp.join(cfg.data.image_dir, phase)
        
        class_folders = [osp.join(folder, label) \
            for label in os.listdir(folder) \
            if osp.isdir(osp.join(folder, label)) \
        ]
        random.shuffle(class_folders)
        
        class_img_dict = {
            osp.basename(f): [osp.join(f, img) for img in os.listdir(f) if (".png" in img or ".jpg" in img)] \
            for f in class_folders
        }
        for k, _ in class_img_dict.items():
            random.shuffle(class_img_dict[k])

        class_list = list(class_img_dict.keys())
        data_list = []
        query_per_class_per_episode = cfg.train.query_per_class_per_episode if phase == "train" else cfg.test.query_per_class_per_episode
        temp_class_list = deepcopy(class_list)
        temp_class_img_dict = deepcopy(class_img_dict)
        while len(temp_class_list) >= self.n_way:
            episode = []
            classes = random.sample(temp_class_list, self.n_way)
            for t, c in enumerate(classes):
                imgs_select = []
                for _ in range(self.k_shot + query_per_class_per_episode):
                    imgs_select.append(temp_class_img_dict[c].pop())
                support_x = imgs_select[:self.k_shot]
                query_x = imgs_select[self.k_shot:]
                episode.append({
                    "support_x": support_x,
                    "query_x": query_x,
                    "target": t
                })
            data_list.append(episode)

            for c in classes:
                if len(temp_class_img_dict[c]) < (self.k_shot + query_per_class_per_episode):
                    temp_class_list.remove(c)
        return data_list

    def __len__(self):
        return len(self.data_list)
