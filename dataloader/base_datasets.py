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
import pickle

from .transforms import square_resize_randomcrop

class BaseDataset(data.Dataset):
    def __init__(self, cfg, phase="train", transform=None):
        super().__init__()

        if phase == "train":
            self.n_way = cfg.train.n_way
            self.k_shot = cfg.train.k_shot
        elif phase == "val":
            self.n_way = cfg.val.n_way
            self.k_shot = cfg.val.k_shot
        else:
            self.n_way = cfg.test.n_way
            self.k_shot = cfg.test.k_shot

        self.data_list = self.prepare_data_list(cfg, phase)
        if transform is None:
            self.transform = square_resize_randomcrop(phase)
        else:
            self.transform = transform(phase)

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
        class_list = class_img_dict.keys()

        data_list = []
        query_per_class_per_episode = cfg.train.query_per_class_per_episode if phase == "train" else cfg.test.query_per_class_per_episode
        if phase == "train":
            episode_per_epoch = cfg.train.episode_per_epoch
        elif phase == "val":
            episode_per_epoch = cfg.val.episode
        else:
            episode_per_epoch = cfg.test.episode

        for e in range(episode_per_epoch):
            episode = []
            classes = random.sample(class_list, self.n_way)
            for t, c in enumerate(classes):
                imgs_set = class_img_dict[c]
                imgs_select = random.sample(imgs_set, self.k_shot + query_per_class_per_episode)
                random.shuffle(imgs_select)
                support_x = imgs_select[:self.k_shot]
                query_x = imgs_select[self.k_shot:]

                episode.append({
                    "support_x": support_x,
                    "query_x": query_x,
                    "target": t
                })
            data_list.append(episode)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        episode = self.data_list[index]
        support_x, support_y, query_x, query_y = [], [], [], []
        for e in episode:
            query_ = e["query_x"]
            for q in query_:
                im = self.transform(Image.open(q).convert("RGB"))
                query_x.append(im.unsqueeze(0))
            support_ = e["support_x"]
            for s in support_:
                im = self.transform(Image.open(s).convert("RGB"))
                support_x.append(im.unsqueeze(0))
            target = e["target"]
            support_y.extend(np.tile(target, len(support_)))
            query_y.extend(np.tile(target, len(query_)))

        support_x = torch.cat(support_x, 0)
        query_x = torch.cat(query_x, 0)
        support_y = torch.LongTensor(support_y)
        query_y = torch.LongTensor(query_y)

        randperm = torch.randperm(len(query_y))
        query_x = query_x[randperm]
        query_y = query_y[randperm]
        return support_x, support_y, query_x, query_y

    def save_summary_datalist(self, save_dir):
        # saving datalist to summary for better reproducing results
        with open(save_dir, 'wb') as f:
            pickle.dump(self.data_list, f)

    def load_summary_datalist(self, load_dir):
        # only for reproducing report results
        with open(load_dir, "rb") as f:
            self.data_list = pickle.load(f)
