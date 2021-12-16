import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from .linear import Linear
from .mel_utils import MELMask

@registry.Query.register("LinearMEL")
class LinearMEL(Linear):
    def __init__(self, in_channels, cfg):
        super().__init__(in_channels, cfg)
        """
        @inproceedings{chen2019closerfewshot,
            title={A Closer Look at Few-shot Classification},
            author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang and  Huang, Jia-Bin},
            booktitle={International Conference on Learning Representations},
            year={2019}
        }

        https://github.com/wyharveychen/CloserLookFewShot
        """

        self.mel_mask = MELMask(cfg)

    def pooling(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        query_mel, support_mel = self.mel_mask(support_xf, query_xf, n_way, k_shot)

        b, q, c, h, w = query_xf.shape
        assert b == 1
        query_xf = (query_xf * query_mel).view(b, q, c, h * w).sum(-1).squeeze(0)
        query_xf = query_xf.clone().detach()

        b, s, c, h, w = support_xf.shape
        assert s == self.n_way * self.k_shot
        assert b == 1
        support_xf = F.adaptive_avg_pool2d(support_xf.squeeze(0), 1).squeeze(-1).squeeze(-1) # [s, c]
        support_xf = support_xf.clone().detach()
        support_y = support_y.view(s)
        return support_xf, support_y, query_xf, query_y
