import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import _l2norm, l2distance

class Similarity(nn.Module):
    def __init__(self, cfg, metric='cosine'):
        super().__init__()
        self.metric = metric

    def forward(self, support_xf, query_xf):
        # query_xf: [b, q, c, h, w]
        # support_xf: [b, n, c, hxw]

        if query_xf.dim() == 5:
            b, q, c, h, w = query_xf.shape
            query_xf = query_xf.view(b, q, c, h*w)
        else:
            b, q = query_xf.shape[:2]

        s = support_xf.shape[1]

        support_xf = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1)
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, s, -1, -1)
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
            return -dist.view(b, q, s, M_q, M_s) / 2.
        else:
            raise NotImplementedError
