import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from time import gmtime, strftime
from tensorboardX import SummaryWriter

import random
import tqdm
from modules.pretrainer import make_pretrain_model
from modules.fsl_query import make_fsl
from dataloader import make_predataloader
from utils import mean_confidence_interval, AverageMeter, set_seed

class Pretrainer(object):
    def __init__(self, cfg, checkpoint_dir):

        self.prefix = osp.basename(checkpoint_dir)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.prefix = osp.basename(checkpoint_dir)
        self.writer = SummaryWriter(self._prepare_summary_snapshots(self.prefix, cfg))

        self.epochs = cfg.pre.epochs

        self.model = make_pretrain_model(cfg).to(self.device)

        self.lr = cfg.pre.lr
        self.lr_decay = cfg.pre.lr_decay
        self.lr_decay_milestones = cfg.pre.lr_decay_milestones

        self.optim = SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=cfg.train.sgd_mom, 
            weight_decay=cfg.train.sgd_weight_decay,
            nesterov=True
        )
        self.lr_scheduler = MultiStepLR(self.optim, milestones=self.lr_decay_milestones, gamma=self.lr_decay)
        self.fsl = make_fsl(cfg).to(self.device)

        self.cfg = cfg
        self.cfg.val.episode = cfg.pre.val_episode

        self.snapshot_epoch = cfg.pre.snapshot_epoch
        self.snapshot_interval = cfg.pre.snapshot_interval

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
        filename = "e0_pre.pth" if postfix is None else "e0_pre_{}.pth".format(postfix)
        filename = osp.join(self.checkpoint_dir, filename)
        state = {
            'fsl': self.fsl.state_dict()
        }
        torch.save(state, filename)

    def train(self, dataloader, epoch):
        losses = AverageMeter()
        tqdm_gen = tqdm.tqdm(dataloader, ncols=80)
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
        tqdm_gen = tqdm.tqdm(dataloader, ncols=80)
        query_y = torch.arange(self.cfg.n_way).repeat(self.cfg.test.query_per_class_per_episode)
        query_y = query_y.type(torch.LongTensor).to(self.device)
        for episode, batch in enumerate(tqdm_gen):
            batch, _ = [b.to(self.device) for b in batch]
            support_x, query_x = batch[:self.cfg.n_way].unsqueeze(0), batch[self.cfg.n_way:].unsqueeze(0)
            support_y = None
            rewards = self.model.forward_test(support_x, support_y, query_x, query_y)
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
        set_seed(1)
        dataloader = make_predataloader(self.cfg, phase="train", batch_size=self.cfg.pre.batch_size)
        val_dataloader = make_predataloader(self.cfg, phase="val")
        for epoch in range(self.epochs):
            epoch_log = epoch + 1
            
            loss_train = self.train(dataloader, epoch_log)
            self.writer.add_scalar('loss_train', loss_train, epoch_log)
            self.lr_scheduler.step()

            if epoch_log >= self.snapshot_epoch and epoch_log % self.snapshot_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    test_accuracy, h = self.validate(val_dataloader)

                self.writer.add_scalar('acc_val', test_accuracy, epoch_log)
                mesg = "\t Testing epoch {} validation accuracy: {:.3f}".format(epoch_log, test_accuracy)
                print(mesg)
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    self.save_model()
                    best_epoch = epoch_log

                self.save_model(postfix=epoch_log)
                print("Current best epoch: {}, accuracy: {:.3f}".format(best_epoch, best_accuracy))

                self.model.train()
