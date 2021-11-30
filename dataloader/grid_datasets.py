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

class GridDataset(BaseDataset):
    def __init__(self, cfg, phase="train", transform=None):
        super().__init__(cfg, phase, transform)

        self.forward_encoding = cfg.model.forward_encoding
        self.pyramid_list = self._parse_encoding_params()

        self.grid_ratio_default = 2.0
        self.phase = phase

    def _parse_encoding_params(self):
        idx = self.forward_encoding.find('-')
        if idx < 0:
            return []
        blocks = self.forward_encoding[idx + 1:].split(',')
        blocks = [int(s) for s in blocks]
        return blocks

    def get_grid_location(self, size, ratio, num_grid):
        '''
        :param size: size of the height/width
        :param ratio: generate grid size/ even divided grid size
        :param num_grid: number of grid
        :return: a list containing the coordinate of the grid
        '''
        raw_grid_size = int(size / num_grid)
        enlarged_grid_size = int(size / num_grid * ratio)

        center_location = raw_grid_size // 2

        location_list = []
        for i in range(num_grid):
            location_list.append((max(0, center_location - enlarged_grid_size // 2),
                                  min(size, center_location + enlarged_grid_size // 2)))
            center_location = center_location + raw_grid_size

        return location_list

    def get_pyramid(self, img, num_grid):
        if self.phase == 'train':
            grid_ratio = 1 + 2 * random.random()
        else:
            grid_ratio = self.grid_ratio_default
        w, h = img.size
        grid_locations_w = self.get_grid_location(w, grid_ratio, num_grid)
        grid_locations_h = self.get_grid_location(h, grid_ratio, num_grid)

        patches_list=[]
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w=grid_locations_w[j]
                patch_location_h=grid_locations_h[i]
                left_up_corner_w=patch_location_w[0]
                left_up_corner_h=patch_location_h[0]
                right_down_cornet_w=patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch=img.crop((left_up_corner_w,left_up_corner_h,right_down_cornet_w,right_down_cornet_h))
                patch=self.transform(patch)
                patches_list.append(patch)
        return patches_list

    def _get_griditems(self, img):
        pyramid_list = []
        for num_grid in self.pyramid_list:
            patches = self.get_pyramid(img, num_grid)
            pyramid_list.extend(patches)
        pyramid_list = torch.cat(pyramid_list, dim=0)
        return pyramid_list

    def __getitem__(self, index):
        episode = self.data_list[index]
        support_x, support_y, query_x, query_y = [], [], [], []
        for e in episode:
            query_ = e["query_x"]
            for q in query_:
                im = self._get_griditems(Image.open(q).convert("RGB"))
                query_x.append(im.unsqueeze(0))
            support_ = e["support_x"]
            for s in support_:
                im = self._get_griditems(Image.open(s).convert("RGB"))
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
