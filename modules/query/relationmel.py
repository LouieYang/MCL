import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from .mel_utils import MELMask

class RelationHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, 1)

        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                n = l.kernel_size[0] * l.kernel_size[1] * l.out_channels
                torch.nn.init.normal_(l.weight, 0, math.sqrt(2. / n))
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)
            elif isinstance(l, nn.Linear):
                torch.nn.init.normal_(l.weight, 0, 0.01)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

@registry.Query.register("RelationMEL")
class RelationMEL(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.cfg = cfg
        self.rn = RelationHead(in_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.mel_mask = MELMask(cfg, gamma=cfg.model.relationnet.mel_gamma, gamma2=cfg.model.relationnet.mel_gamma2)

        if cfg.model.relationnet.mel_mask == "query":
            self.score_func = self._scores_query # by default
        elif cfg.model.relationnet.mel_mask == "support":
            self.score_func = self._scores_support
        elif cfg.model.relationnet.mel_mask == "both":
            self.score_func = self._scores_both
        else:
            raise NotImplementedError

    def _scores_query(self, support_xf, support_y, query_xf, query_y, query_mel, support_mel):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w)
        support_xf = support_xf.mean(2) # [b, N, c, h, w]
        # [b*N, c]

        support_xf = F.adaptive_avg_pool2d(support_xf.view(-1, c, h, w), 1).squeeze(-1).squeeze(-1) 
        support_xf = support_xf.view(b, -1, c)
        support_xf = support_xf.unsqueeze(1).expand(-1, q, -1, -1)

        # [b*q, c]
        query_xf = (query_xf * query_mel).view(b, q, c, h * w).sum(-1)
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, self.n_way, -1)

        comb = torch.cat((support_xf, query_xf), 3).view(-1, 2 * c)
        scores = self.rn(comb).view(-1, self.n_way)
        return scores

    def _scores_support(self, support_xf, support_y, query_xf, query_y, query_mel, support_mel):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w)
        support_xf = support_xf.mean(2) # [b, N, c, h, w]
        # [b*N, c]

        support_mel = support_mel.view(b, q, self.n_way, -1, h, w)
        support_mel = support_mel / (support_mel.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1))

        support_xf = support_xf.unsqueeze(1).repeat(1, q, 1, 1, 1, 1) * support_mel
        support_xf = support_xf.sum(-1).sum(-1)

        # [b*q, c]
        query_xf = F.adaptive_avg_pool2d(query_xf.view(-1, c, h, w), 1).squeeze(-1).squeeze(-1)
        query_xf = query_xf.view(b, -1, c) 
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, self.n_way, -1)

        comb = torch.cat((support_xf, query_xf), 3).view(-1, 2 * c)
        scores = self.rn(comb).view(-1, self.n_way)
        return scores

    def _scores_both(self, support_xf, support_y, query_xf, query_y, query_mel, support_mel):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w)
        support_xf = support_xf.mean(2) # [b, N, c, h, w]
        # [b*N, c]

        support_mel = support_mel.view(b, q, self.n_way, -1, h, w)
        support_mel = support_mel / (support_mel.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1))

        support_xf = support_xf.unsqueeze(1).repeat(1, q, 1, 1, 1, 1) * support_mel
        support_xf = support_xf.sum(-1).sum(-1)

        # [b*q, c]
        query_xf = (query_xf * query_mel).view(b, q, c, h * w).sum(-1)
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, self.n_way, -1)

        comb = torch.cat((support_xf, query_xf), 3).view(-1, 2 * c)
        scores = self.rn(comb).view(-1, self.n_way)
        return scores


    def __call__(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot
        query_mel, support_mel = self.mel_mask(support_xf, query_xf, n_way, k_shot)
        scores = self.score_func(support_xf, support_y, query_xf, query_y, query_mel, support_mel)
        N = scores.shape[0]
        query_y = query_y.view(N)
        if self.training:
            loss = self.criterion(scores, query_y)
            return {"reltion_loss": loss}
        else:
            _, predict_labels = torch.max(scores, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(N)]
            return rewards
