import os
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data

from .base_datasets import BaseDataset
from .grid_datasets import GridDataset
from .miniimagenet import MiniImagenet
from .tieredimagenet import TieredImagenet
from .distributed_sampler import DistributedSampler
from .pretrain_datasets import PreDataset
from .samplers import CategoriesSampler

def _decide_dataset(cfg, phase):
    if cfg.model.forward_encoding.startswith("Grid"):
        return GridDataset(cfg, phase)
    if cfg.model.forward_encoding.startswith("Sampling"):
        return SamplingDataset(cfg, phase)

    data_folder = osp.basename(osp.abspath(cfg.data.image_dir))
    if data_folder == "miniImagenet":
        dataset = MiniImagenet(cfg, phase)
    elif data_folder == "tieredimagenet":
        dataset = TieredImagenet(cfg, phase)
    else:
        raise NotImplementedError
    return dataset

def make_dataloader(cfg, phase, batch_size=1):
    dataset = _decide_dataset(cfg, phase)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    return dataloader

def make_predataloader(cfg, phase, batch_size=1):
    dataset = PreDataset(cfg, phase)
    if phase == "train":
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
        )
    elif phase == "val":
        sampler = CategoriesSampler(
            dataset.label, 
            cfg.pre.val_episode, cfg.n_way, 
            1 + cfg.test.query_per_class_per_episode
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, num_workers=8, pin_memory=True
        )
    else:
        raise NotImplementedError
    return dataloader

def make_distributed_dataloader(cfg, phase, batch_size, distributed_info, epoch=0):
    dataset = _decide_dataset(cfg, phase)
    sampler = DistributedSampler(
        dataset, 
        num_replicas=distributed_info["num_replicas"],
        rank=distributed_info["rank"],
        shuffle=True
    )
    sampler.set_epoch(epoch)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_sampler=batch_sampler
    )
    return dataloader
