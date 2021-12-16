import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from modules.utils import batched_index_select
from .similarity import Similarity

@registry.Query.register("DMN4")
class DMN4(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.cfg = cfg
        self.inner_simi = Similarity(cfg, metric='cosine')
        self.criterion = nn.CrossEntropyLoss()

        self.temperature = cfg.model.dmn4.temperature
        self.k_shot_average = cfg.model.dmn4.larger_shot == "average"
        self.is_norm = cfg.model.dmn4.is_norm
        if not self.is_norm:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        device = support_xf.device
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        if self.k_shot_average:
            support_xf = support_xf.view(b, n_way, k_shot, c, h, w).mean(2)
            support_xf = support_xf.view(b, n_way, c, h * w)

        S = self.inner_simi(support_xf, query_xf)
        M_q, M_s = S.shape[-2:]
        S_class_merged = S.permute(0, 1, 3, 2, 4).contiguous().view(b, q, M_q, -1)

        query_nearest = S_class_merged.max(-1)[1]
        query_class_diff = torch.topk(S.max(-1)[0], 2, 2)[0]
        query_class_diff = query_class_diff[:, :, 0, :] - query_class_diff[:, :, 1, :] # [b, q, M_q]
        diffs_m = torch.nn.functional.one_hot(query_nearest, n_way * M_s).float() * query_class_diff.unsqueeze(-1)
        diffs_m_max, diffs_max_nearest = diffs_m.max(-2)

        diff_mask = batched_index_select(diffs_max_nearest.view(-1, n_way * M_s).unsqueeze(1), 2, query_nearest.view(-1, M_q))
        diff_mask = (diff_mask == torch.arange(M_q, device=device).expand_as(diff_mask)).view(b, q, M_q).float()
        if self.is_norm:
            nMNN = diff_mask.sum(-1).view(-1)
            diff_mask = diff_mask / diff_mask.sum(-1, keepdim=True)
        diff_mask = diff_mask * self.temperature

        predict = (S.max(-1)[0] * diff_mask.unsqueeze(2)).sum(-1)
        predict = predict.view(b*q, n_way)

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
