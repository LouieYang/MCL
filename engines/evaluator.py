import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

import random
from modules.fsl_query import make_fsl
from dataloader import make_dataloader
from engines.utils import mean_confidence_interval, AverageMeter, set_seed

class evaluator(object):
    def __init__(self, cfg, checkpoint_dir):

        self.n_way                 = cfg.test.n_way # 5
        self.k_shot                = cfg.test.k_shot # 5
        self.test_query_per_class   = cfg.test.query_per_class_per_episode  # 15

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.eval_epoch = osp.basename(checkpoint_dir)
        self.prediction_folder = osp.join(
            "./predictions/", osp.basename(checkpoint_dir[:checkpoint_dir.rfind("/")])
        )
        if not osp.exists(self.prediction_folder):
            os.mkdir(self.prediction_folder)

        self.prediction_dir = osp.join(
            self.prediction_folder,
            "predictions.txt"
        )

        self.checkpoint_dir = checkpoint_dir

        self.fsl = make_fsl(cfg).to(self.device)

        state_dict = torch.load(checkpoint_dir)
        self.fsl.load_state_dict(state_dict["fsl"], strict=False)
        self.fsl.eval()

        self.test_episode = cfg.test.episode
        self.total_testtimes = cfg.test.total_testtimes

        self.cfg = cfg

    def run(self):
        with open(self.prediction_dir, 'w') as f_txt:
            total_accuracies = 0.0
            total_h = 0.0
            print("evaluation epoch: ", self.eval_epoch, file=f_txt)
            set_seed(1)
            for epoch in range(self.total_testtimes):
                test_dataloader = make_dataloader(self.cfg, phase="test", batch_size=self.cfg.test.batch_size)
                tqdm_gen = tqdm(test_dataloader, ncols=80)
                accuracies = []
                acc = AverageMeter()
                for episode, (support_x, support_y, query_x, query_y) in enumerate(tqdm_gen):
                    support_x            = support_x.to(self.device)
                    support_y            = support_y.to(self.device)
                    query_x              = query_x.to(self.device)
                    query_y              = query_y.to(self.device)

                    rewards = self.fsl(support_x, support_y, query_x, query_y, self.n_way, self.k_shot)
                    if isinstance(rewards, tuple):
                        rewards = rewards[0]

                    total_rewards = np.sum(rewards)
                    accuracy = total_rewards / (query_y.numel())
                    acc.update(accuracy, 1)
                    mesg = "Acc={:.4f}".format(acc.avg)
                    tqdm_gen.set_description(mesg)
                    accuracies.append(accuracy)

                test_accuracy, h = mean_confidence_interval(accuracies)
                print("test accuracy:",test_accuracy,"h:",h)
                print("test_accuracy:", test_accuracy, "h:", h, file=f_txt)
                total_accuracies += test_accuracy
                total_h += h
            print("aver_accuracy:", total_accuracies/self.total_testtimes, "h:", total_h/self.total_testtimes)
            print("aver_accuracy:", total_accuracies/self.total_testtimes, "h:", total_h/self.total_testtimes, file=f_txt)
            return test_accuracy
