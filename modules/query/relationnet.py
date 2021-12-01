import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

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

@registry.Query.register("RelationNet")
class RelationNet(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @inproceedings{sung2018learning,
            title={Learning to Compare: Relation Network for Few-Shot Learning},
            author={Sung, Flood and Yang, Yongxin and Zhang, Li and Xiang, Tao and Torr, Philip HS and Hospedales, Timothy M},
            booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
            year={2018}
        }

        https://github.com/floodsung/LearningToCompare_FSL
        """

        self.cfg = cfg
        self.rn = RelationHead(in_channels)
        self.criterion = nn.CrossEntropyLoss()

    def _scores(self, support_xf, support_y, query_xf, query_y):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w)
        support_xf = support_xf.mean(2) # [b, N, c, h, w]
        # [b*N, c]
        support_xf = F.adaptive_avg_pool2d(support_xf.view(-1, c, h, w), 1).squeeze(-1).squeeze(-1) 
        support_xf = support_xf.view(b, -1, c)
        support_xf = support_xf.unsqueeze(1).expand(-1, q, -1, -1)
        # [b*q, c]
        query_xf = F.adaptive_avg_pool2d(query_xf.view(-1, c, h, w), 1).squeeze(-1).squeeze(-1)
        query_xf = query_xf.view(b, -1, c) 
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, self.n_way, -1)

        comb = torch.cat((support_xf, query_xf), 3).view(-1, 2 * c)
        scores = self.rn(comb).view(-1, self.n_way)
        return scores


    def __call__(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        scores = self._scores(support_xf, support_y, query_xf, query_y)
        N = scores.shape[0]
        query_y = query_y.view(N)
        if self.training:
            loss = self.criterion(scores, query_y)
            return {"reltion_loss": loss}
        else:
            _, predict_labels = torch.max(scores, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(N)]
            return rewards
