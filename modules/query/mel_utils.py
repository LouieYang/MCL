import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from modules.utils import _l2norm, batched_index_select

from .similarity import Similarity

class MELMask(nn.Module):

    def __init__(self, cfg, katz_factor=0.999, gamma=20.0, gamma2=10.0):

        super().__init__()

        self.cfg = cfg

        self.inner_simi = Similarity(cfg, metric='cosine')

        self.gamma = gamma
        self.gamma2 = gamma2
        self.katz_factor = katz_factor

    def forward(self, support_xf, query_xf, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        b, s, c, h, w = support_xf.shape
        q = query_xf.shape[1]
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
        support_xf = support_xf.view(b, self.n_way, c, h * w)
        S = self.inner_simi(support_xf, query_xf) # [b, q, N, M_q, M_s]
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        N_examples = b * q
        St = S.transpose(-2, -1)
        device = S.device

        T_sq = torch.exp(self.gamma * (S - S.max(-1, keepdim=True)[0]))
        T_sq = T_sq / T_sq.sum(-1, keepdim=True) # row-wise stochastic
        T_qs = torch.exp(self.gamma2 * (St - St.max(-1, keepdim=True)[0])) # [b * q, M_s, M_q]
        T_qs = T_qs / T_qs.sum(-1, keepdim=True) # row-wise stochastic

        T = torch.cat([
            torch.cat([torch.zeros((N_examples, M_s, M_s), device=device), T_sq.transpose(-2, -1)], dim=-1),
            torch.cat([T_qs.transpose(-2, -1), torch.zeros((N_examples, M_q, M_q), device=device)], dim=-1),
        ], dim=-2)

        katz = (torch.inverse(torch.eye(M_s + M_q, device=device)[None].repeat(N_examples, 1, 1) - self.katz_factor * T) - \
                torch.eye(M_s + M_q, device=S.device)[None].repeat(N_examples, 1, 1))@torch.ones((N_examples, M_s + M_q, 1), device=device)
        katz_query = katz.squeeze(-1)[:, M_s:] / katz.squeeze(-1)[:, M_s:].sum(-1, keepdim=True)
        katz_support = katz.squeeze(-1)[:, :M_s] / katz.squeeze(-1)[:, :M_s].sum(-1, keepdim=True)
        katz_support = katz_support.view(b, q, self.n_way, -1)
        katz_support = katz_support / katz_support.sum(-1, keepdim=True)
        katz_support = katz_support.view(b, q, self.n_way, h, w).unsqueeze(3) # [b, q, self.n_way, c, h, w]
        katz_query = katz_query.view(b, q, h, w).unsqueeze(2)
        return katz_query, katz_support
