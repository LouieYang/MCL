import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.registry as registry
from .base import BasePretrainer
from modules.query.similarity import Similarity

@registry.Pretrainer.register("MEL")
class PretrainMEL(BasePretrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        resolution = cfg.pre.resolution
        self.category_mat = nn.Parameter(
            torch.randn(1, cfg.pre.pretrain_num_class, self.encoder.out_channels, resolution),
            requires_grad=True
        )
        self.num_category = cfg.pre.pretrain_num_class
        self.inner_simi = Similarity(cfg, metric='cosine')

        self.katz_factor = cfg.model.mel.katz_factor
        self.gamma = nn.Parameter(torch.FloatTensor([cfg.model.gamma]),requires_grad=True)
        self.gamma2 = nn.Parameter(torch.FloatTensor([cfg.model.gamma2]),requires_grad=True)
        # self.gamma = cfg.model.mel.gamma
        # self.gamma2 = cfg.model.mel.gamma2

        self.criterion = nn.NLLLoss()

    def forward(self, x, y):
        alpha = self.katz_factor
        alpha_2 = alpha * alpha

        enc = self.encoder(x)
        enc = enc[None] # [1, b, c, h, w]
        b, q = enc.shape[:2]

        S = self.inner_simi(self.category_mat, None, enc, None) # [1, b, cat, res, res]
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        N_examples, M_q, M_s = S.shape
        St = S.transpose(-2, -1)
        device = S.device

        P_sq = torch.exp(self.gamma * (St - St.max(-2, keepdim=True)[0]))
        P_sq = P_sq / P_sq.sum(-2, keepdim=True)
        P_qs = torch.exp(self.gamma2 * (S - S.max(-2, keepdim=True)[0]))
        P_qs = P_qs / P_qs.sum(-2, keepdim=True)

        inverted_matrix = torch.inverse(torch.eye(M_q, device=device)[None].repeat(N_examples, 1, 1) - alpha_2 * P_qs@P_sq)
        katz = (alpha_2 * P_sq@inverted_matrix@P_qs).sum(-1) + (alpha * P_sq@inverted_matrix).sum(-1)
        katz = katz / katz.sum(-1, keepdim=True)
        predict = katz.view(N_examples, self.num_category, -1).sum(-1)

        loss = self.criterion(torch.log(predict), y)
        return {"pretrain_loss": loss}

    def forward_test(self, support_x, support_y, query_x, query_y):
        alpha = self.katz_factor
        alpha_2 = alpha * alpha

        b, q = query_x.shape[:2]
        s = support_x.shape[1]
        
        support_x = support_x.view((-1,) + support_x.shape[-3:])
        support_x = self.encoder(support_x)
        support_xf = support_x.view((b, s) + support_x.shape[-3:])
        query_x = query_x.view((-1,) + query_x.shape[-3:])
        query_x = self.encoder(query_x)
        query_xf = query_x.view((b, q) + query_x.shape[-3:])

        device = support_xf.device

        b, s, c, h, w = support_xf.shape
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
        support_xf = support_xf.view(b, self.n_way, c, h * w)

        S = self.inner_simi(support_xf, support_y, query_xf, query_y) # [b, q, N, M_q, M_s]
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        N_examples, M_q, M_s = S.shape
        St = S.transpose(-2, -1)
        device = S.device

        P_sq = torch.exp(self.gamma * (St - St.max(-2, keepdim=True)[0]))
        P_sq = P_sq / P_sq.sum(-2, keepdim=True)
        P_qs = torch.exp(self.gamma2 * (S - S.max(-2, keepdim=True)[0]))
        P_qs = P_qs / P_qs.sum(-2, keepdim=True)

        inverted_matrix = torch.inverse(torch.eye(M_q, device=device)[None].repeat(N_examples, 1, 1) - alpha_2 * P_qs@P_sq)
        katz = (alpha_2 * P_sq@inverted_matrix@P_qs).sum(-1) + (alpha * P_sq@inverted_matrix).sum(-1)
        katz = katz / katz.sum(-1, keepdim=True)
        predicts = katz.view(N_examples, self.n_way, -1).sum(-1)

        query_y = query_y.view(N_examples)
        _, predict_labels = torch.max(predicts, 1)
        rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
        return rewards

