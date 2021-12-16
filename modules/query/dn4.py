import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from modules.utils import _l2norm
from .similarity import Similarity

@registry.Query.register("DN4")
class DN4(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @inproceedings{li2019DN4,
            title={Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning},
            author={Li, Wenbin and Wang, Lei and Xu, Jinglin and Huo, Jing and Gao Yang and Luo, Jiebo},
            booktitle={CVPR},
            year={2019}
        }
        https://github.com/WenbinLee/DN4
        """

        self.cfg = cfg
        self.neighbor_k = 1

        self.inner_simi = Similarity(cfg, metric='cosine')
        self.criterion = nn.CrossEntropyLoss()
        self.k_shot_average = cfg.model.dn4.larger_shot == "average"

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]
        if self.k_shot_average:
            support_xf = support_xf.view(b, n_way, k_shot, c, h, w).mean(2)
            support_xf = support_xf.view(b, n_way, c, h * w)

        innerproduct_matrix = self.inner_simi(support_xf, query_xf)
        topk_value, _ = torch.topk(innerproduct_matrix, self.neighbor_k, -1) # [b, q, N, M_q, neighbor_k]
        similarity_matrix = topk_value.mean(-1).view(b, q, n_way, -1).sum(-1)
        similarity_matrix = similarity_matrix.view(b * q, n_way)

        query_y = query_y.view(b * q)
        if self.training:
            loss = self.criterion(similarity_matrix, query_y)
            return {"dn4_loss": loss}
        else:
            _, predict_labels = torch.max(similarity_matrix, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
