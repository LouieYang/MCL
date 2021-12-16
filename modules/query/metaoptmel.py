import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from .metaoptnet import MetaOptNet
from .mel_utils import MELMask

@registry.Query.register("MetaOptMEL")
class MetaOptMEL(MetaOptNet):
    
    def __init__(self, in_channels, cfg):
        super().__init__(in_channels, cfg)
        """
        @inproceedings{lee2019meta,
            title={Meta-Learning with Differentiable Convex Optimization},
            author={Kwonjoon Lee and Subhransu Maji and Avinash Ravichandran and Stefano Soatto},
            booktitle={CVPR},
            year={2019}
        }

        https://github.com/kjunelee/MetaOptNet/
        """

        self.mel_mask = MELMask(cfg)

    def pooling(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        query_mel, support_mel = self.mel_mask(support_xf, query_xf, n_way, k_shot)

        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        query = (query_xf * query_mel).view(b, q, c, h * w).sum(-1)
        support = F.adaptive_avg_pool2d(support_xf.view(-1, c, h, w), 1).view(b, s, c)
        return query, support

