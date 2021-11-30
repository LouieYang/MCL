import re
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from time import gmtime, strftime
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import numpy as np
from tensorboardX import SummaryWriter

import random
import tqdm
from modules.fsl_query import make_fsl
from dataloader import make_dataloader
from engines.utils import mean_confidence_interval, AverageMeter, set_seed

class trainer(object):
    def __init__(self, cfg, checkpoint_dir):

        self.seed = cfg.seed
        set_seed(self.seed) # should set seed for training from scratch with Conv4 backbone

        self.n_way                 = cfg.train.n_way # 5
        self.k_shot                = cfg.train.k_shot # 5
        self.train_query_per_class = cfg.train.query_per_class_per_episode # 10
        self.val_query_per_class   = cfg.val.query_per_class_per_episode  # 15
        self.val_interval          = cfg.val.interval
        self.train_episode_per_epoch = cfg.train.episode_per_epoch
        self.prefix = osp.basename(checkpoint_dir)
        self.save_train_datalist = cfg.train.save_train_datalist
        self.writer_dir = self._prepare_summary_snapshots(self.prefix, cfg)
        self.writer = SummaryWriter(self.writer_dir)
        if self.save_train_datalist:
            os.makedirs(osp.join(self.writer_dir, "datalist"))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.epochs = cfg.train.epochs
        self.epoch_start_val = cfg.val.epoch_start_val

        self.fsl = make_fsl(cfg).to(self.device)

        self.lr = cfg.train.learning_rate
        self.lr_decay = cfg.train.lr_decay
        self.lr_decay_epoch = cfg.train.lr_decay_epoch
        if cfg.train.optim == "Adam":
            self.optim = Adam(self.fsl.parameters(),lr=cfg.train.learning_rate, betas=cfg.train.adam_betas)
        elif cfg.train.optim == "SGD":
            self.optim = SGD(
                self.fsl.parameters(), 
                lr=cfg.train.learning_rate, 
                momentum=cfg.train.sgd_mom, 
                weight_decay=cfg.train.sgd_weight_decay,
                nesterov=True
            )
        else:
            raise NotImplementedError
        pths = [osp.basename(f) for f in glob.glob(osp.join(checkpoint_dir, "*.pth")) if "best" not in f]
        if pths:
            pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
            pths = [p for p, e in zip(pths, pths_epoch) if e]
            pths_epoch = [int(e) for e in pths_epoch if e]
            self.train_start_epoch = max(pths_epoch)
            c = osp.join(checkpoint_dir, pths[pths_epoch.index(self.train_start_epoch)])
            state_dict = torch.load(c)
            self.fsl.load_state_dict(state_dict["fsl"], strict=False)
            print("[*] Continue training from checkpoints: {}".format(c))
            lr_scheduler_last_epoch = self.train_start_epoch
            if "optimizer" in state_dict and state_dict["optimizer"] is not None:
                self.optim.load_state_dict(state_dict["optimizer"])
        else:
            self.train_start_epoch = 0
            lr_scheduler_last_epoch = -1

        if cfg.train.lr_decay_milestones:
            self.lr_scheduler = MultiStepLR(self.optim, milestones=cfg.train.lr_decay_milestones,gamma=self.lr_decay)
        elif cfg.train.lr_scheduler == "CosineAnnealingLR":
            self.lr_scheduler = CosineAnnealingLR(self.optim, T_max=self.epochs - self.train_start_epoch, last_epoch=lr_scheduler_last_epoch)
        elif cfg.train.lr_scheduler == "StepLR":
            self.lr_scheduler = StepLR(self.optim, step_size=self.lr_decay_epoch, gamma=self.lr_decay)
        else:
            raise NotImplementedError

        self.snapshot_name = lambda prefix: \
            osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.pth".format(prefix, self.n_way, self.k_shot))
        self.snapshot_record = lambda prefix: \
            osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.txt".format(prefix, self.n_way, self.k_shot))
        self.cfg = cfg

    def _prepare_summary_snapshots(self, prefix, cfg):
        summary_prefix = osp.join(cfg.train.summary_snapshot_base, prefix)
        summary_dir = osp.join(summary_prefix, strftime("%Y-%m-%d-%H:%M", gmtime()))
        for d_ in [summary_prefix, summary_dir]:
            if not osp.exists(d_):
                os.mkdir(d_)
        return summary_dir

    def fix_bn(self):
        for module in self.fsl.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.SyncBatchNorm):
                module.eval()

    def validate(self, dataloader):
        accuracies = []
        tqdm_gen = tqdm.tqdm(dataloader, ncols=80, leave=False)
        acc = AverageMeter()
        loss_meter = AverageMeter()
        for episode, (support_x, support_y, query_x, query_y) in enumerate(tqdm_gen):
            support_x            = support_x.to(self.device)
            support_y            = support_y.to(self.device)
            query_x              = query_x.to(self.device)
            query_y              = query_y.to(self.device)

            rewards = self.fsl(support_x, support_y, query_x, query_y, self.cfg.val.n_way, self.cfg.val.k_shot)
            if isinstance(rewards, tuple):
                rewards, losses = rewards
                loss_meter.update(losses.item(), len(query_x))
            total_rewards = np.sum(rewards)

            accuracy = total_rewards / (query_y.numel())

            acc.update(total_rewards / query_y.numel(), 1)
            mesg = "Val: acc={:.4f}".format(
                acc.avg
            )
            tqdm_gen.set_description(mesg)

            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        return test_accuracy, h, loss_meter

    def save_model(self, prefix, accuracy, h, epoch, final_epoch=False):
        filename = self.snapshot_name(prefix)
        recordname = self.snapshot_record(prefix)
        state = {
            'summary_dir': osp.basename(self.writer_dir),
            'episode': prefix,
            'fsl': self.fsl.state_dict(),
            'epoch': epoch,
            # "optimizer": None if not final_epoch else self.optim.state_dict()
        }
        with open(recordname, 'w') as f:
            f.write("prefix: {}\nepoch: {}\naccuracy: {}\nh: {}\n".format(prefix, epoch, accuracy, h)) 
        if int(re.search(r'([\d.]+)', torch.__version__).group(1).replace('.', '')) > 160:
            torch.save(state, filename, _use_new_zipfile_serialization=False) # compatible with early torch versions to load
        else:
            torch.save(state, filename)

    def train(self, dataloader, epoch):
        losses = AverageMeter()
        tqdm_gen = tqdm.tqdm(dataloader, ncols=80, leave=False)

        self.optim.zero_grad()
        for episode, (support_x, support_y, query_x, query_y) in enumerate(tqdm_gen):
            support_x            = support_x.to(self.device)
            support_y            = support_y.to(self.device)
            query_x              = query_x.to(self.device)
            query_y              = query_y.to(self.device)

            loss = self.fsl(support_x, support_y, query_x, query_y, self.cfg.train.n_way, self.cfg.train.k_shot)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.update(loss.item(), len(query_x))
            mesg = "epoch {}, loss={:.3f}".format(
                epoch, 
                losses.avg
            )
            tqdm_gen.set_description(mesg)
        return losses.avg

    def run(self):
        print("[={}=]".format(self.prefix))
        best_accuracy = 0.0
        best_accuracy_epoch = 0
        set_seed(self.seed)
        val_dataloader = make_dataloader(
            self.cfg, 
            phase="val", batch_size=self.cfg.test.batch_size,
            save_summary_dir = osp.join(self.writer_dir, "datalist", "val.pkl") if self.save_train_datalist else None
        )

        tqdm_gen = tqdm.tqdm(range(self.train_start_epoch, self.epochs), ncols=80)
        for epoch in tqdm_gen:
            train_dataloader = make_dataloader(
                self.cfg, phase="train", 
                batch_size=self.cfg.train.batch_size,
                save_summary_dir = osp.join(self.writer_dir, "datalist", "train_epoch_{}".format(epoch)) if self.save_train_datalist else None
            )
            loss_train = self.train(train_dataloader, epoch + 1)
            self.writer.add_scalar('loss_train', loss_train, epoch + 1)

            if epoch < self.epoch_start_val or (self.val_interval > 0 and (epoch + 1) % self.val_interval != 0):
                self.lr_scheduler.step()
                continue

            self.fsl.eval()
            with torch.no_grad():
                val_accuracy, h, val_loss_meter = self.validate(val_dataloader)
                if val_loss_meter.count > 0:
                    self.writer.add_scalar('loss_val', val_loss_meter.avg, epoch + 1)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_accuracy_epoch = epoch + 1
                self.save_model("best", val_accuracy, h, epoch + 1, True)
            # mesg = "\t Testing epoch {} validation accuracy: {:.4f}, h: {:.3f}".format(epoch + 1, val_accuracy, h)
            # print(mesg)
            self.writer.add_scalar('acc_val', val_accuracy, epoch + 1)

            self.lr_scheduler.step()
            if self.cfg.train.checkpoint_interval > 0 and epoch + 1 % self.cfg.train.checkpoint_interval == 0:
                self.save_model(epoch + 1, val_accuracy, h, epoch + 1)

            self.fsl.train()
            if self.cfg.train.fix_bn:
                self.fix_bn()

            mesg = "loss/val: {:.3f}/{:.4f}, best val: {:.4f}({})".format(loss_train, val_accuracy, best_accuracy, best_accuracy_epoch)
            tqdm_gen.set_description(mesg)
