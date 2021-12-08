import re
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from time import gmtime, strftime
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tensorboardX import SummaryWriter
import copy

import random

from modules.fsl_query import make_fsl
from dataloader import make_distributed_dataloader

from engines.utils import mean_confidence_interval, AverageMeter, set_seed
from engines.distributed_utils import get_world_size, reduce_loss_dict

import tqdm

class DistributedTrainer(object):
    def __init__(self, args, cfg, checkpoint_dir):
        ## Append for distributed training
        self.distributed = args.distributed
        assert self.distributed
        self.base_rank = args.base_rank
        self.local_rank = args.local_rank 
        self.rank = self.base_rank + self.local_rank
        self.verbose = (self.rank == 0)
        self.seed = cfg.seed

        set_seed(self.seed, self.verbose) # should set seed for training from scratch with Conv4 backbone

        self.device = torch.device("cuda")

        self.train_n_way                 = cfg.train.n_way # 5
        self.train_k_shot                = cfg.train.k_shot # 5
        self.val_n_way                 = cfg.val.n_way # 5
        self.val_k_shot                = cfg.val.k_shot # 5
        self.train_query_per_class = cfg.train.query_per_class_per_episode # 10
        self.val_query_per_class   = cfg.test.query_per_class_per_episode  # 15
        self.train_episode_per_epoch = cfg.train.episode_per_epoch
        self.prefix = osp.basename(checkpoint_dir)
        self.writer_dir = self._prepare_summary_snapshots(self.prefix, cfg)
        self.writer = SummaryWriter(self.writer_dir) if self.verbose else None
        self.lr = cfg.train.learning_rate
        self.lr_decay = cfg.train.lr_decay
        self.lr_decay_epoch = cfg.train.lr_decay_epoch
        self.checkpoint_dir = checkpoint_dir
        self.epochs = cfg.train.epochs

        fsl = make_fsl(cfg)
        pths = [osp.basename(f) for f in glob.glob(osp.join(checkpoint_dir, "*.pth")) if "best" not in f]
        if pths:
            pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
            pths = [p for p, e in zip(pths, pths_epoch) if e]
            pths_epoch = [int(e) for e in pths_epoch if e]
            self.train_start_epoch = max(pths_epoch)
            c = osp.join(checkpoint_dir, pths[pths_epoch.index(self.train_start_epoch)])
            state_dict = torch.load(c, map_location=torch.device('cpu'))
            fsl.load_state_dict(state_dict["fsl"])
            if self.verbose:
                print("[*] Continue training from checkpoints: {}".format(c))
            lr_scheduler_last_epoch = self.train_start_epoch
            # if "optimizer" in state_dict and state_dict["optimizer"] is not None:
            #    self.optim.load_state_dict(state_dict["optimizer"])
        else:
            self.train_start_epoch = 0
            lr_scheduler_last_epoch = -1

        fsl = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fsl).to(self.device)
        if cfg.train.optim == "Adam":
            self.optim = Adam(fsl.parameters(),lr=cfg.train.learning_rate, betas=cfg.train.adam_betas)
        elif cfg.train.optim == "SGD":
            self.optim = SGD(
                fsl.parameters(), 
                lr=cfg.train.learning_rate, 
                momentum=cfg.train.sgd_mom, 
                weight_decay=cfg.train.sgd_weight_decay,
                nesterov=True
            )
        else:
            raise NotImplementedError

        self.cfg = cfg
        self.fsl = torch.nn.parallel.DistributedDataParallel(
            fsl, device_ids=[self.local_rank], output_device=self.local_rank,
        )
        if cfg.train.lr_decay_milestones:
            self.lr_scheduler = MultiStepLR(self.optim, milestones=cfg.train.lr_decay_milestones,gamma=self.lr_decay)
        else:
            self.lr_scheduler = StepLR(self.optim, step_size=self.lr_decay_epoch, gamma=self.lr_decay)

        self.snapshot_name = lambda prefix: \
            osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.pth".format(prefix, self.train_n_way, self.train_k_shot))
        self.snapshot_record = lambda prefix: \
            osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.txt".format(prefix, self.train_n_way, self.train_k_shot))

        self.best_state_dict_for_distributed = None

    def _prepare_summary_snapshots(self, prefix, cfg):
        if self.verbose:
            summary_prefix = osp.join(cfg.train.summary_snapshot_base, prefix)
            summary_dir = osp.join(summary_prefix, strftime("%Y-%m-%d-%H:%M", gmtime()))
            for d_ in [summary_prefix, summary_dir]:
                if not osp.exists(d_):
                    os.mkdir(d_)
            return summary_dir
        else:
            return None

    def fix_bn(self):
        for module in self.fsl.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.SyncBatchNorm):
                module.eval()

    def validate(self, dataloader):
        accuracies = []
        if self.verbose:
            dataloader = tqdm.tqdm(dataloader, ncols=80, leave=False)
            acc = AverageMeter()
        for episode, (support_x, support_y, query_x, query_y) in enumerate(dataloader):
            support_x            = support_x.to(self.device)
            support_y            = support_y.to(self.device)
            query_x              = query_x.to(self.device)
            query_y              = query_y.to(self.device)

            rewards = self.fsl(support_x, support_y, query_x, query_y, self.val_n_way, self.val_k_shot)
            if isinstance(rewards, tuple):
                rewards, _ = rewards
            total_rewards = np.sum(rewards)
            accuracy = total_rewards / (query_y.numel())
            if self.verbose:
                acc.update(total_rewards / query_y.numel(), 1)
                mesg = "Val: acc={:.3f}".format(acc.avg)
                dataloader.set_description(mesg)

            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        return test_accuracy, h

    def save_model(self, prefix, accuracy, h, epoch, final_epoch=False):
        filename = self.snapshot_name(prefix)
        recordname = self.snapshot_record(prefix)
        state = {
            'episode': prefix,
            'fsl': self.fsl.module.state_dict(),
            'epoch': epoch,
            "optimizer": None if not final_epoch else self.optim.state_dict()
        }
        with open(recordname, 'w') as f:
            f.write("prefix: {}\nepoch: {}\naccuracy: {}\nh: {}\n".format(prefix, epoch, accuracy, h)) 
        torch.save(state, filename)

    def train(self, dataloader, epoch):
        losses = AverageMeter()
        if self.verbose:
            dataloader = tqdm.tqdm(dataloader, ncols=80, leave=False)

        for episode, (support_x, support_y, query_x, query_y) in enumerate(dataloader):
            support_x            = support_x.to(self.device)
            support_y            = support_y.to(self.device)
            query_x              = query_x.to(self.device)
            query_y              = query_y.to(self.device)

            loss = self.fsl(support_x, support_y, query_x, query_y, self.train_n_way, self.train_k_shot)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses_dict_reduced = reduce_loss_dict({"loss": loss})
            losses_reduced = sum(losses_dict_reduced.values())
            losses.update(losses_reduced.item(), len(query_x))
            if self.verbose:
                mesg = "epoch {}, loss={:.3f}".format(epoch, losses.avg)
                dataloader.set_description(mesg)
        return losses.avg

    def run(self):
        best_accuracy = 0.0
        best_accuracy_epoch = 0
        set_seed(self.seed, self.verbose)
        val_dataloader = make_distributed_dataloader(
            self.cfg, phase="val", 
            batch_size=self.cfg.test.batch_size // get_world_size(),
            distributed_info={"num_replicas": get_world_size(), "rank": self.rank}
        )

        tqdm_gen = range(self.train_start_epoch, self.epochs)
        if self.verbose:
            tqdm_gen = tqdm.tqdm(tqdm_gen, ncols=80)
        for epoch in tqdm_gen:
            train_dataloader = make_distributed_dataloader(
                self.cfg, 
                phase="train", 
                batch_size=self.cfg.train.batch_size // get_world_size(),
                distributed_info={"num_replicas": get_world_size(), "rank": self.rank},
                epoch=epoch
            )
            loss_train = self.train(train_dataloader, epoch + 1)
            if self.verbose:
                self.writer.add_scalar('loss_train', loss_train, epoch + 1)

            self.fsl.eval()
            with torch.no_grad():
                total_accuracies, total_h = self.validate(val_dataloader)
                validation_results = {
                    "acc": torch.Tensor([total_accuracies]).to('cuda'), 
                    "h": torch.Tensor([total_h]).to('cuda')
                }
            validation_results_reduced = reduce_loss_dict(validation_results)
            total_accuracies = validation_results_reduced["acc"].item()
            total_h = validation_results_reduced["h"].item()

            if self.verbose:
                mesg = "loss/val: {:.3f}/{:.4f}, best val: {:.4f}({})".format(loss_train, total_accuracies, best_accuracy, best_accuracy_epoch)
                tqdm_gen.set_description(mesg)
                self.writer.add_scalar('acc_val', total_accuracies, epoch + 1)

            if total_accuracies > best_accuracy:
                best_accuracy = total_accuracies
                best_accuracy_epoch = epoch + 1
                self.best_state_dict_for_distributed = copy.deepcopy(self.fsl.state_dict())

                if self.verbose:
                    self.save_model("best", total_accuracies, total_h, epoch + 1, True)

            self.lr_scheduler.step()
            self.fsl.train()
            if self.cfg.train.fix_bn:
                self.fix_bn()
