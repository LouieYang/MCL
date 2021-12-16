import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from modules.utils import _l2norm, batched_index_select

from .similarity import Similarity

@registry.Query.register("LS")
class LabelSpreading(nn.Module):
    # Label Spreading
    def __init__(self, in_channels, cfg, metric='cosine'):
        super().__init__()

        self.cfg = cfg
        self.metric = metric

        self.inner_simi = Similarity(cfg, metric='cosine')
        self.gamma = cfg.model.ls.gamma

        self.criterion = nn.NLLLoss()
        self.factor = 0.99

    def averaging_based_similarities(self, support_xf, support_y, query_xf, query_y):
        b, s, c, h, w = support_xf.shape
        q = query_xf.shape[1]
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
        support_xf = support_xf.view(b, self.n_way, c, h * w)
        S = self.inner_simi(support_xf, query_xf) # [b, q, N, M_q, M_s]
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        return S

    def _inner_state_similarity(self, states, mask_diagonal=None):
        device = states.device
        b, n, c, h, w = states.shape
        states = _l2norm(states.view(b * n, c, h * w), dim=1)
        inner_simi = states.transpose(-2, -1)@states
        if mask_diagonal:
            mask = torch.eye(h * w, h * w, device=device)[None]
            inner_simi = (1 - mask) * inner_simi + mask * mask_diagonal
        return inner_simi.view(b, n, h * w, h * w)

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        S = self.averaging_based_similarities(support_xf, support_y, query_xf, query_y) # [N, M_q, M_s]
        N_examples, M_q, M_s = S.shape
        b, q, c, h, w = query_xf.shape
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2).permute(0, 2, 1, 3, 4).contiguous()
        support_xf = support_xf.view(b, c, self.n_way, h * w).unsqueeze(1)
        support_inner = self._inner_state_similarity(support_xf).repeat(1, q, 1, 1)
        support_inner = support_inner.view(N_examples, M_s, M_s)
        query_inner = self._inner_state_similarity(query_xf).view(N_examples, M_q, M_q) # [N, M_q, M_q]

        W = torch.cat([
            torch.cat([support_inner, S.transpose(-2, -1)], dim=-1),
            torch.cat([S, query_inner], dim=-1),
        ], dim=-2)
        mask = torch.eye(M_q + M_s, device = S.device).unsqueeze(0).repeat(N_examples, 1, 1)
        W = torch.exp(self.gamma * W) * (1 - mask)

        D = torch.diag_embed(torch.sqrt(1. / W.sum(-1)))
        W = D@W@D
        Y = torch.cat([torch.eye(self.n_way, device=S.device).repeat(1, h * w).view(-1, self.n_way), torch.zeros((M_q, self.n_way), device=S.device)], dim=0)
        Y = Y[None].repeat(N_examples, 1, 1)
        F = torch.inverse((torch.eye(M_q + M_s, device=S.device)[None].repeat(N_examples, 1, 1) - self.factor * W))@Y # [N_examples, M_s + M_q, self.n_way]

        # option 1: example-wise classification
        example_F = F.view(N_examples, -1, h * w, self.n_way).mean(2)[:, -1, :]
        example_F = example_F / example_F.sum(-1, keepdim=True)
        query_y = query_y.view(N_examples)
        loss = self.criterion(torch.log(example_F), query_y)
        if self.training:
            return {"LS_loss": loss}
        else:
            _, predict_labels = torch.max(example_F, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(N_examples)]
            return rewards, loss
