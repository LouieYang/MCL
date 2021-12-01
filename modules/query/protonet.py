import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from modules.utils import l2distance

@registry.Query.register("ProtoNet")
class ProtoNet(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @inproceedings{snell2017prototypical,
            title={Prototypical networks for few-shot learning},
            author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
            booktitle={NIPS},
            pages={4077--4087},
            year={2017}
        }

        https://github.com/Sha-Lab/FEAT/blob/master/model/models/protonet.py
        """

        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()

        self.temperature = cfg.model.protonet.temperature

    def _scores(self, support_xf, support_y, query_xf, query_y):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        #support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w)
        support_xf = support_xf.view((-1,) + support_xf.shape[-3:])
        support_xf = F.adaptive_avg_pool2d(support_xf, 1).view(b, self.n_way, self.k_shot, c)
        support_proto = support_xf.mean(-2) # [b, self.n_way, c]

        query_xf = F.adaptive_avg_pool2d(query_xf.view(-1, c, h, w), 1).view(b, q, c)
        scores = -l2distance(query_xf.transpose(-2, -1).contiguous(), support_proto.transpose(-2, -1).contiguous())
        scores = scores.view(b * q, -1)
        return scores

    def __call__(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot
        scores = self._scores(support_xf, support_y, query_xf, query_y)
        N = scores.shape[0]
        query_y = query_y.view(N)
        if self.training:
            loss = self.criterion(scores / self.temperature, query_y)
            return {"protonet": loss}
        else:
            _, predict_labels = torch.max(scores, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(N)]
            return rewards

