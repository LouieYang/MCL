import os
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data
from functools import partial

from .base_datasets import BaseDataset
from .grid_datasets import GridDataset
from .unrepeated_datasets import UnrepeatedDataset
from .distributed_sampler import DistributedSampler
from .pretrain_datasets import PreDataset
from .samplers import CategoriesSampler
from .transforms import square_resize_randomcrop, reflectpad_randomcrop

def _decide_dataset(cfg, phase, save_summary_dir=None, load_summary_dir=None):
    data_folder = osp.basename(osp.abspath(cfg.data.image_dir))
    if data_folder == "meta_iNat" or data_folder == "tiered_meta_iNat":
        t = partial(reflectpad_randomcrop, image_size=cfg.data.image_size, pad_size=cfg.data.pad_size)
    else:
        t = partial(square_resize_randomcrop, image_size=cfg.data.image_size, pad_size=cfg.data.pad_size)

    if phase != "train" or cfg.train.episode_first_dataloader:
        if cfg.model.forward_encoding.startswith("Grid"):
            dataset = GridDataset(cfg, phase, t)
        else:
            dataset = BaseDataset(cfg, phase, t)
    else:
        dataset = UnrepeatedDataset(cfg, phase, t)

    if save_summary_dir is not None:
        dataset.save_summary_datalist(save_summary_dir)
    if load_summary_dir is not None:
        dataset.save_summary_datalist(load_summary_dir)
    return dataset

def make_dataloader(cfg, phase, batch_size=1, save_summary_dir=None, load_summary_dir=None):
    dataset = _decide_dataset(cfg, phase, save_summary_dir, load_summary_dir)
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
            cfg.pre.val_episode, cfg.val.n_way, 
            1 + cfg.val.query_per_class_per_episode
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, num_workers=8, pin_memory=True
        )
    else:
        raise NotImplementedError
    return dataloader

def make_distributed_dataloader(cfg, phase, batch_size, distributed_info, epoch=0, pretrain=False):
    if pretrain:
        dataset = PreDataset(cfg, phase)
    else:
        dataset = _decide_dataset(cfg, phase, None, None)

    if pretrain and phase == "val":
        batch_sampler = CategoriesSampler(
            dataset.label,
            cfg.pre.val_episode, cfg.val.n_way,
            1 + cfg.val.query_per_class_per_episode
        )
    else:
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

