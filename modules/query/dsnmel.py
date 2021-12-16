import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from .dsn import DSN
from .mel_utils import MELMask

@registry.Query.register("DSNMEL")
class DSNMEL(DSN):
    def __init__(self, in_channels, cfg):
        super().__init__(in_channels, cfg)
        """
        @inproceedings{simon2020dsn,
            author       = {C. Simon}, {P. Koniusz}, {R. Nock}, and {M. Harandi}
            title        = {Adaptive Subspaces for Few-Shot Learning},
            booktitle    = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
            year         = 2020
        }

        https://github.com/chrysts/dsn_fewshot
        """

        self.mel_mask = MELMask(cfg)

    def pooling(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        query_mel, support_mel = self.mel_mask(support_xf, query_xf, n_way, k_shot)

        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        query = (query_xf * query_mel).view(b, q, c, h * w).sum(-1)
        support = F.adaptive_avg_pool2d(support_xf.view(-1, c, h, w), 1).view(b, s, c)
        return query, support

