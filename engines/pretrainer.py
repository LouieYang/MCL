import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
from time import gmtime, strftime
from tensorboardX import SummaryWriter

import random
import tqdm
from modules.pretrainer import make_pretrain_model
from modules.fsl_query import make_fsl
from dataloader import make_predataloader
from engines.utils import mean_confidence_interval, AverageMeter, set_seed, GradualWarmupScheduler

class Pretrainer(object):
    def __init__(self, cfg, checkpoint_dir):

        self.seed = cfg.seed

        self.prefix = osp.basename(checkpoint_dir)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.prefix = osp.basename(checkpoint_dir)
        self.dataset_prefix = osp.basename(cfg.data.image_dir).lower()
        self.writer_dir = self._prepare_summary_snapshots(self.prefix, cfg)
        self.writer = SummaryWriter(self.writer_dir)

        self.epochs = cfg.pre.epochs

        self.model = make_pretrain_model(cfg).to(self.device)

        self.lr = cfg.pre.lr
        self.lr_decay = cfg.pre.lr_decay
        self.lr_decay_milestones = cfg.pre.lr_decay_milestones

        self.optim = SGD(
            [{'params': self.model.parameters(), 'initial_lr': self.lr}], 
            lr=self.lr, 
            momentum=cfg.train.sgd_mom, 
            weight_decay=cfg.train.sgd_weight_decay,
            nesterov=True
        )
        pths = [osp.basename(f) for f in glob.glob(osp.join(checkpoint_dir, "*_pre_all.pth")) if "best" not in f]
        if pths:
            pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
            pths = [p for p, e in zip(pths, pths_epoch) if e]
            pths_epoch = [int(e) for e in pths_epoch if e]
            self.train_start_epoch = max(pths_epoch)
            c = osp.join(checkpoint_dir, pths[pths_epoch.index(self.train_start_epoch)])
            state_dict = torch.load(c)
            self.model.load_state_dict(state_dict)
            print("[*] Continue training from checkpoints: {}".format(c))
            lr_scheduler_last_epoch = self.train_start_epoch
            if "optimizer" in state_dict and state_dict["optimizer"] is not None:
                self.optim.load_state_dict(state_dict["optimizer"])
        else:
            self.train_start_epoch = 0
            lr_scheduler_last_epoch = -1

        if cfg.pre.lr_scheduler == "CosineAnnealingLR":
            lr_scheduler = CosineAnnealingLR(self.optim, T_max=cfg.pre.epochs, last_epoch=lr_scheduler_last_epoch)
        else:
            lr_scheduler = MultiStepLR(self.optim, milestones=self.lr_decay_milestones, gamma=self.lr_decay, last_epoch=lr_scheduler_last_epoch)
        if cfg.pre.warmup_scheduler_epoch > 0:
            self.lr_scheduler = GradualWarmupScheduler(self.optim, multiplier=1, total_epoch=cfg.pre.warmup_scheduler_epoch, after_scheduler=lr_scheduler)
        else:
            self.lr_scheduler = lr_scheduler
        self.fsl = make_fsl(cfg).to(self.device)

        self.cfg = cfg
        self.cfg.val.episode = cfg.pre.val_episode

        self.snapshot_epoch = cfg.pre.snapshot_epoch
        self.snapshot_interval = cfg.pre.snapshot_interval
        self.snapshot_for_meta = "{}-e0_pre.pth".format(self.dataset_prefix)

    def _prepare_summary_snapshots(self, prefix, cfg):
        summary_prefix = osp.join(cfg.train.summary_snapshot_base, prefix)
        summary_dir = osp.join(summary_prefix, strftime("%Y-%m-%d-%H:%M", gmtime()))
        for d_ in [summary_prefix, summary_dir]:
            if not osp.exists(d_):
                os.mkdir(d_)
        return summary_dir

    def save_model(self, postfix=None):
        self.fsl.encoder.load_state_dict(self.model.encoder.state_dict())
        self.fsl.eval()
        filename_for_meta = self.snapshot_for_meta if postfix is None else "e{}_pre.pth".format(postfix)
        filename_all = filename_for_meta.replace("pre", "pre_all")
        filename_for_meta = osp.join(self.checkpoint_dir, filename_for_meta)
        filename_all = osp.join(self.checkpoint_dir, filename_all)
        state_for_meta = {
            'fsl': self.fsl.state_dict()
        }
        state_all = self.model.state_dict()
        torch.save(state_for_meta, filename_for_meta)
        torch.save(state_all, filename_all)

    def train(self, dataloader, epoch):
        losses = AverageMeter()
        tqdm_gen = tqdm.tqdm(dataloader, ncols=80, leave=False)
        for iters, (x, y) in enumerate(tqdm_gen):
            x = x.to(self.device)
            y = y.to(self.device)

            loss = self.model(x, y)
            loss_sum = sum(loss.values())

            self.optim.zero_grad()
            loss_sum.backward()
            self.optim.step()
            losses.update(loss_sum.item(), len(y))

            mesg = "epoch {}, loss={:.3f}".format(
                epoch, 
                losses.avg
            )
            tqdm_gen.set_description(mesg)
        return losses.avg

    def validate(self, dataloader):
        accuracies = []
        acc = AverageMeter()
        tqdm_gen = tqdm.tqdm(dataloader, ncols=80, leave=False)
        query_y = torch.arange(self.cfg.val.n_way).repeat(self.cfg.val.query_per_class_per_episode)
        query_y = query_y.type(torch.LongTensor).to(self.device)
        for episode, batch in enumerate(tqdm_gen):
            batch, _ = [b.to(self.device) for b in batch]
            support_x, query_x = batch[:self.cfg.val.n_way].unsqueeze(0), batch[self.cfg.val.n_way:].unsqueeze(0)
            support_y = None
            rewards = self.model(support_x, support_y, query_x, query_y)
            total_rewards = np.sum(rewards)

            accuracy = total_rewards / (query_y.numel())
            acc.update(total_rewards / query_y.numel(), 1)
            mesg = "Val: acc={:.3f}".format(
                acc.avg
            )
            tqdm_gen.set_description(mesg)
            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        return test_accuracy, h

    def run(self):
        best_accuracy = 0.0
        best_epoch = -1
        set_seed(self.seed)
        dataloader = make_predataloader(self.cfg, phase="train", batch_size=self.cfg.pre.batch_size)
        val_dataloader = make_predataloader(self.cfg, phase="val")
        tqdm_gen = tqdm.tqdm(range(self.train_start_epoch, self.epochs), ncols=80)
        for epoch in tqdm_gen:
            epoch_log = epoch + 1
            
            loss_train = self.train(dataloader, epoch_log)
            self.writer.add_scalar('loss_train', loss_train, epoch_log)
            self.lr_scheduler.step()

            if epoch_log >= self.snapshot_epoch and epoch_log % self.snapshot_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    test_accuracy, h = self.validate(val_dataloader)

                self.writer.add_scalar('acc_val', test_accuracy, epoch_log)
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    self.save_model()
                    best_epoch = epoch_log

                mesg = "loss/val: {:.3f}/{:.4f}, best val: {:.4f}({})".format(loss_train, test_accuracy, best_accuracy, best_epoch)
                tqdm_gen.set_description(mesg)
                self.model.train()
