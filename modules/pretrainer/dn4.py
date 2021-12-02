import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.registry as registry
from .base import BasePretrainer
from modules.query.similarity import Similarity

@registry.Pretrainer.register("DN4")
class PretrainDN4(BasePretrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        resolution = cfg.pre.resolution
        self.category_mat = nn.Parameter(
            torch.randn(1, cfg.pre.pretrain_num_class, self.encoder.out_channels, resolution),
            requires_grad=True
        )
        self.num_category = cfg.pre.pretrain_num_class
        self.inner_simi = Similarity(cfg, metric='cosine')
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)

    def forward_train(self, x, y):
        enc = self.encoder(x)
        enc = enc[None] # [1, b, c, h, w]

        S = self.inner_simi(self.category_mat, None, enc, None) # [1, b, cat, res, res]
        predict = S.max(-1)[0].sum(-1)
        predict = predict.view(-1, self.num_category)
        predict = predict * self.scale

        if self.training:
            loss = self.criterion(predict, y)
            return {"pretrain_loss": loss}
        else:
            _, predict_labels = torch.max(predict, 1)
            rewards = [1 if predict_labels[j]==y[j].to(predict_labels.device) else 0 for j in range(len(y))]
            return rewards

    def forward_test(self, support_x, support_y, query_x, query_y):
        b, q = query_x.shape[:2]
        s = support_x.shape[1]

        
        support_x = support_x.view((-1,) + support_x.shape[-3:])
        support_x = self.encoder(support_x)
        support_x = support_x.view((b, s) + support_x.shape[-3:])
        query_x = query_x.view((-1,) + query_x.shape[-3:])
        query_x = self.encoder(query_x)
        query_x = query_x.view((b, q) + query_x.shape[-3:])

        b, s, c, h, w = support_x.shape
        support_x = support_x.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
        support_x = support_x.view(b, self.n_way, c, h * w)

        innerproduct_matrix = self.inner_simi(support_x, support_y, query_x, query_y)
        topk_value, _ = torch.topk(innerproduct_matrix, 1, -1) # [b, q, N, M_q, neighbor_k]
        similarity_matrix = topk_value.mean(-1).view(b, q, self.n_way, -1).sum(-1)
        similarity_matrix = similarity_matrix.view(b * q, self.n_way)

        query_y = query_y.view(b * q)
        _, predict_labels = torch.max(similarity_matrix, 1)
        rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
        return rewards

