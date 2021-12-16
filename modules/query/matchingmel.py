import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from modules.utils import _l2norm

from .mel_utils import MELMask

@registry.Query.register("MatchingMEL")
class MatchingMEL(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()
        self.mel_mask = MELMask(cfg, gamma=cfg.model.matchingnet.mel_gamma, gamma2=cfg.model.matchingnet.mel_gamma2)

        if cfg.model.matchingnet.mel_mask == "query":
            self.score_func = self._scores_query # by default
        elif cfg.model.matchingnet.mel_mask == "support":
            self.score_func = self._scores_support
        elif cfg.model.matchingnet.mel_mask == "both":
            self.score_func = self._scores_both
        else:
            raise NotImplementedError

        self.temperature = cfg.model.matchingnet.temperature

    def _scores_query(self, support_xf, support_y, query_xf, query_y, query_mel, support_mel):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        support_xf = support_xf.view((-1,) + support_xf.shape[-3:])
        support_xf = F.adaptive_avg_pool2d(support_xf, 1).view(b, self.n_way, self.k_shot, c)
        support_xf = support_xf.mean(-2) # [b, self.n_way, c]

        query_xf = (query_xf * query_mel).view(b, q, c, -1).sum(-1)
        
        support_proto = _l2norm(support_xf, dim=-1)
        query_xf = _l2norm(query_xf, dim=-1)

        scores = query_xf@support_proto.transpose(-2, -1)
        scores = scores.view(b * q, -1)
        return scores

    def _scores_support(self, support_xf, support_y, query_xf, query_y, query_mel, support_mel):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w).mean(2).unsqueeze(1)
        support_xf = support_xf.repeat(1, q, 1, 1, 1, 1) * support_mel
        support_xf = support_xf.view(b, q, self.n_way, c, -1).sum(-1)

        query_xf = query_xf.unsqueeze(2).repeat(1, 1, self.n_way, 1, 1, 1).mean(-1).mean(-1)
        
        support_proto = _l2norm(support_xf, dim=-1)
        query_xf = _l2norm(query_xf, dim=-1)

        scores = (query_xf * support_proto).sum(-1).view(b * q, -1)
        return scores

    def _scores_both(self, support_xf, support_y, query_xf, query_y, query_mel, support_mel):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w).mean(2).unsqueeze(1)
        support_xf = support_xf.repeat(1, q, 1, 1, 1, 1) * support_mel
        support_xf = support_xf.view(b, q, self.n_way, c, -1).sum(-1)

        query_xf = (query_xf * query_mel).view(b, q, c, -1).sum(-1)
        
        support_proto = _l2norm(support_xf, dim=-1)
        query_xf = _l2norm(query_xf, dim=-1)

        scores = (query_xf.unsqueeze(-2) * support_proto).sum(-1).view(b * q, -1)
        return scores

    def __call__(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        query_mel, support_mel = self.mel_mask(support_xf, query_xf, n_way, k_shot)
        scores = self.score_func(support_xf, support_y, query_xf, query_y, query_mel, support_mel)
        N = scores.shape[0]
        query_y = query_y.view(N)
        if self.training:
            loss = self.criterion(scores * self.temperature, query_y)
            return {"MatchingMEL": loss}
        else:
            _, predict_labels = torch.max(scores, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(N)]
            return rewards
