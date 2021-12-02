import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.registry as registry
from .base import BasePretrainer

@registry.Pretrainer.register("Linear")
class PretrainLinear(BasePretrainer):
    def __init__(self, cfg):

        super().__init__(cfg)
        
        self.fake_classifier = nn.Linear(self.encoder.out_channels, cfg.pre.pretrain_num_class)
        self.avg_pool = nn.AvgPool2d(int(math.sqrt(cfg.pre.resolution)), stride=1)

    def forward_train(self, x, y):
        enc = self.encoder(x)
        enc = self.avg_pool(enc).squeeze(-1).squeeze(-1)
        out = self.fake_classifier(enc)
        if self.training:
            loss = self.criterion(out, y)
            return {"pretrain_loss": loss}
        else:
            _, predict_labels = torch.max(out, 1)
            rewards = [1 if predict_labels[j]==y[j].to(predict_labels.device) else 0 for j in range(len(y))]
            return rewards

    def forward_test(self, support_x, support_y, query_x, query_y):
        b, q = query_x.shape[:2]

        support_x = support_x.view((-1,) + support_x.shape[-3:])
        support_x = self.avg_pool(self.encoder(support_x)).squeeze(-1).squeeze(-1)
        query_x = query_x.view((-1,) + query_x.shape[-3:])
        query_x = self.avg_pool(self.encoder(query_x)).squeeze(-1).squeeze(-1)

        proto = support_x.view(b, self.n_way, self.k_shot, -1).mean(2) # [b, N, -1]
        proto = F.normalize(proto, p=2, dim=-1).transpose(-2, -1)
        query = query_x.view(b, q, -1) # [b, q, -1]

        similarity_matrix = (query@proto).view(b * q, -1) # [b, q, N]
        query_y = query_y.view(b * q)

        _, predict_labels = torch.max(similarity_matrix, 1)
        rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
        return rewards

