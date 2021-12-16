import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from .negative_cosine import NegativeCosine
from .mel_utils import MELMask

@registry.Query.register("NegativeCosineMEL")
class NegativeCosineMEL(NegativeCosine):
    def __init__(self, in_channels, cfg):
        super().__init__(in_channels, cfg)
        """
        @inproceedings{liu2020negative,
            title={Negative margin matters: Understanding margin in few-shot classification},
            author={Liu, Bin and Cao, Yue and Lin, Yutong and Li, Qi and Zhang, Zheng and Long, Mingsheng and Hu, Han},
            booktitle={European Conference on Computer Vision},
            pages={438--455},
            year={2020},
            organization={Springer}
        }

        https://github.com/bl0/negative-margin.few-shot
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
