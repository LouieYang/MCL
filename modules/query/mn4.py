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
        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot

        self.inner_simi = Similarity(cfg, metric='cosine')
        self.criterion = nn.CrossEntropyLoss()

        self.temperature = cfg.model.mn4.temperature
        self.k_shot_average = cfg.model.mn4.larger_shot == "average"

    def _1MNN_mask(self, simi_matrix):
        b, q, N, M_q, M_s = simi_matrix.shape
        simi_matrix_merged = simi_matrix.permute(0, 1, 3, 2, 4).contiguous().view(b, q, M_q, -1)
        query_nearest = simi_matrix_merged.max(-1)[1]

        class_wise_max = (simi_matrix.max(-1)[0]).max(2)[0] + 1
        class_m = torch.nn.functional.one_hot(query_nearest, self.n_way * M_s).float() * class_wise_max.unsqueeze(-1)
        class_m_max, support_nearest = class_m.max(-2)
        support_nearest[class_m_max == 0] = M_q + 1 # Compensation for single connection

        # [b, q, M_q]
        mask = batched_index_select(support_nearest.view(-1, self.n_way * M_s).unsqueeze(1), 2, query_nearest.view(-1, M_q))
        mask = (mask == torch.arange(M_q, device=simi_matrix.device).expand_as(mask)).view(b, q, M_q)
        return mask

    def forward(self, support_xf, support_y, query_xf, query_y):
        device = support_xf.device
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        if self.k_shot_average:
            support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
            support_xf = support_xf.view(b, self.n_way, c, h * w)
        else:
            support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).permute(0, 1, 3, 2, 4, 5)
            support_xf = support_xf.contiguous().view(b, self.n_way, c, -1)

        S = self.inner_simi(support_xf, support_y, query_xf, query_y)
        query_mask = self._1MNN_mask(S).float() 
        query_mask = query_mask * self.temperature

        predict = (S.max(-1)[0] * query_mask.unsqueeze(2)).sum(-1)
        predict = predict.view(b*q, self.n_way)
        query_y = query_y.view(b * q)

        if self.training:
            loss = self.criterion(predict, query_y)
            return {"loss": loss}
        else:
            _, predict_labels = torch.max(predict, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards

