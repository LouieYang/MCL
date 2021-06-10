import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from modules.utils import _l2norm
from .innerproduct_similarity import InnerproductSimilarity

@registry.Query.register("DN4")
class DN4(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.neighbor_k = cfg.model.nbnn_topk

        self.inner_simi = InnerproductSimilarity(cfg, metric='cosine')
        self.temperature = cfg.model.temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, support_xf, support_y, query_xf, query_y):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        innerproduct_matrix = self.inner_simi(support_xf, support_y, query_xf, query_y)
        topk_value, _ = torch.topk(innerproduct_matrix, self.neighbor_k, -1) # [b, q, N, M_q, neighbor_k]
        similarity_matrix = topk_value.mean(-1).view(b, q, self.n_way, -1).sum(-1)
        similarity_matrix = similarity_matrix.view(b * q, self.n_way)

        query_y = query_y.view(b * q)
        if self.training:
            loss = self.criterion(similarity_matrix / self.temperature, query_y)
            return {"dn4_loss": loss}
        else:
            _, predict_labels = torch.max(similarity_matrix, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
