import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from modules.utils import batched_index_select
from .similarity import Similarity

@registry.Query.register("MN4")
class MN4(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.cfg = cfg
        self.inner_simi = Similarity(cfg, metric='cosine')

        self.temperature = cfg.model.mn4.temperature
        self.k_shot_average = cfg.model.mn4.larger_shot == "average"
        self.is_norm = cfg.model.mn4.is_norm

        if not self.is_norm:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _1MNN_mask(self, S):
        b, q, N, M_q, M_s = S.shape
        v, idx = S.max(-1)
        v, query_nearest = v.max(2)
        query_nearest = query_nearest * M_s + torch.gather(idx, 2, query_nearest.unsqueeze(2)).squeeze(2)

        # assert non-negative
        v = torch.nn.functional.one_hot(query_nearest, self.n_way * M_s).float() * (v.unsqueeze(-1) + 1)
        v, support_nearest = v.max(-2)
        support_nearest[v == 0] = M_q + 1 # Compensation for single connection

        # [b, q, M_q]
        mask = batched_index_select(support_nearest.view(-1, self.n_way * M_s).unsqueeze(1), 2, query_nearest.view(-1, M_q))
        mask = (mask == torch.arange(M_q, device=S.device).expand_as(mask)).view(b, q, M_q)
        return mask

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        device = support_xf.device
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        if self.k_shot_average:
            support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
            support_xf = support_xf.view(b, self.n_way, c, h * w)
        S = self.inner_simi(support_xf, query_xf)
        query_mask = self._1MNN_mask(S).float() 
        if self.is_norm:
            nMNN = query_mask.sum(-1).view(-1)
            query_mask = query_mask / query_mask.sum(-1, keepdim=True)

        query_mask = query_mask * self.temperature
        predict = (S.max(-1)[0] * query_mask.unsqueeze(2)).sum(-1)
        predict = predict.view(b*q, self.n_way)
        query_y = query_y.view(b * q)
        if self.training:
            if not self.is_norm:
                loss = self.criterion(predict, query_y)
            else:
                nMNN = nMNN / nMNN.sum()
                loss = (self.criterion(predict, query_y) * nMNN).sum()
            return {"loss": loss}
        else:
            _, predict_labels = torch.max(predict, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
