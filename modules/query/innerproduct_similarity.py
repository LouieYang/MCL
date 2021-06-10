import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import _l2norm, l2distance

class InnerproductSimilarity(nn.Module):
    def __init__(self, cfg, metric='cosine'):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.metric = metric

    def forward(self, support_xf, support_y, query_xf, query_y):
        if support_xf.dim() == 5:
            b, s, c, h, w = support_xf.shape
            support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).permute(0, 1, 3, 2, 4, 5)
            support_xf = support_xf.contiguous().view(b, self.n_way, c, -1)
        #else: deleted for semi, if FUTURE failed, please check this
        #    b, s, c, k = support_xf.shape
        #    support_xf = support_xf.view(b, self.n_way, self.k_shot, c, k).permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_way, c, -1)

        if query_xf.dim() == 5:
            b, q, c, h, w = query_xf.shape
            query_xf = query_xf.view(b, q, c, h*w)

        support_xf = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1)
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, self.n_way, -1, -1)
        M_q = query_xf.shape[-1]
        M_s = support_xf.shape[-1]

        if self.metric == 'cosine':
            support_xf = _l2norm(support_xf, dim=-2)
            query_xf = _l2norm(query_xf, dim=-2)
            query_xf = torch.transpose(query_xf, 3, 4)
            return query_xf@support_xf # bxQxNxM_qxM_s
        elif self.metric == 'innerproduct':
            query_xf = torch.transpose(query_xf, 3, 4)
            return query_xf@support_xf # bxQxNxM_qxM_s
        elif self.metric == 'euclidean':
            return 1 - l2distance(query_xf, support_xf)
        elif self.metric == 'neg_ed':
            query_xf = query_xf.contiguous().view(-1, c, M_q).transpose(-2, -1).contiguous()
            support_xf = support_xf.contiguous().view(-1, c, M_s).transpose(-2, -1).contiguous()
            dist = torch.cdist(query_xf, support_xf)
            return -dist.view(b, q, self.n_way, M_q, M_s) / 2.
        else:
            raise NotImplementedError
