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
from dataloader import make_distributed_dataloader

from engines.utils import mean_confidence_interval, AverageMeter, set_seed
from engines.distributed_utils import reduce_loss_dict, get_world_size, synchronize

class DistributedEvaluator(object):
    def __init__(self, args, cfg, checkpoint_dir, fsl=None):
        self.seed = cfg.seed

        self.distributed = args.distributed
        assert self.distributed
        self.base_rank = args.base_rank
        self.local_rank = args.local_rank 
        self.rank = self.base_rank + self.local_rank
        self.device = torch.device("cuda")
        self.verbose = (self.rank == 0)

        self.n_way                 = cfg.test.n_way # 5
        self.k_shot                = cfg.test.k_shot # 5
        self.test_query_per_class   = cfg.test.query_per_class_per_episode  # 15

        self.eval_epoch = osp.basename(checkpoint_dir)
        if self.verbose:
            self.prediction_folder = osp.join(
                "./predictions/", osp.basename(checkpoint_dir[:checkpoint_dir.rfind("/")])
            )
            if not osp.exists(self.prediction_folder):
                os.mkdir(self.prediction_folder)

            self.prediction_dir = osp.join(
                self.prediction_folder,
                "predictions.txt"
                # osp.basename(checkpoint_dir).replace(".pth", ".txt")
            )

        self.checkpoint_dir = checkpoint_dir
        if fsl is None:
            fsl = make_fsl(cfg).to(self.device)
            state_dict = torch.load(checkpoint_dir)
            fsl.load_state_dict(state_dict, strict=False)
            self.fsl = torch.nn.parallel.DistributedDataParallel(
                fsl, device_ids=[self.local_rank], output_device=self.local_rank,
            )
        else:
            self.fsl = fsl
        self.fsl.eval()

        self.test_episode = cfg.test.episode
        self.total_testtimes = cfg.test.total_testtimes

        self.cfg = cfg

    def run(self):
        if self.verbose:
            f_txt = open(self.prediction_dir, 'w')

        total_accuracies = 0.0
        set_seed(self.seed)
        for epoch in range(self.total_testtimes):
            test_dataloader = make_distributed_dataloader(
                self.cfg, phase="test", batch_size=self.cfg.test.batch_size // get_world_size(),
                distributed_info={"num_replicas": get_world_size(), "rank": self.rank},
                epoch=epoch
            )

            accuracies = []
            if self.verbose:
                tqdm_gen = tqdm(test_dataloader, ncols=80)
                acc = AverageMeter()
            else:
                tqdm_gen = test_dataloader
            for episode, (support_x, support_y, query_x, query_y) in enumerate(tqdm_gen):
                support_x            = support_x.to(self.device)
                support_y            = support_y.to(self.device)
                query_x              = query_x.to(self.device)
                query_y              = query_y.to(self.device)

                rewards = self.fsl(support_x, support_y, query_x, query_y)
                if isinstance(rewards, tuple):
                    rewards = rewards[0]

                total_rewards = np.sum(rewards)
                accuracy = total_rewards / (query_y.numel())
                if self.verbose:
                    acc.update(accuracy, 1)
                    mesg = "Acc={:.4f}".format(acc.avg)
                    tqdm_gen.set_description(mesg)

                accuracies.append(accuracy)
            test_accuracy, h = mean_confidence_interval(accuracies)
            test_results = {"acc": torch.Tensor([test_accuracy]).to('cuda')}
            test_results_reduced = reduce_loss_dict(test_results)

            total_accuracies += test_results_reduced["acc"].item()

        if self.verbose:
            print("aver_accuracy:", total_accuracies/self.total_testtimes)
            print("aver_accuracy:", total_accuracies/self.total_testtimes, file=f_txt)
            f_txt.close()
        return total_accuracies
