import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

@registry.Query.register("DSN")
class DSN(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @inproceedings{simon2020dsn,
            author       = {C. Simon}, {P. Koniusz}, {R. Nock}, and {M. Harandi}
            title        = {Adaptive Subspaces for Few-Shot Learning},
            booktitle    = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
            year         = 2020
        }

        https://github.com/chrysts/dsn_fewshot
        """

        self.cfg = cfg

        self.d = in_channels
        self.criterion = nn.CrossEntropyLoss()

    def pooling(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        query = F.adaptive_avg_pool2d(query_xf.view(-1, c, h, w), 1).view(b, q, c)
        support = F.adaptive_avg_pool2d(support_xf.view(-1, c, h, w), 1).view(b, s, c)
        return query, support

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):

        assert k_shot > 1

        self.n_way = n_way
        self.k_shot = k_shot

        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        query, support = self.pooling(support_xf, support_y, query_xf, query_y, n_way, k_shot)

        tasks_per_batch = query.size(0)
        n_support = support.size(1)
        n_query = query.size(1)
        d = query.size(2)

        support_labels_one_hot = F.one_hot(support_y.view(tasks_per_batch * n_support), self.n_way).float()
        support_reshape = support.view(tasks_per_batch * n_support, -1)
        support_labels_reshaped = support_y.contiguous().view(-1)
        class_representatives = []
        for nn in range(self.n_way):
            idxss = (support_labels_reshaped == nn).nonzero()
            all_support_perclass = support_reshape[idxss, :]
            class_representatives.append(all_support_perclass.view(tasks_per_batch, k_shot, -1))
        class_representatives = torch.stack(class_representatives)
        class_representatives = class_representatives.transpose(0, 1) #tasks_per_batch, n_way, n_support, -1
        class_representatives = class_representatives.transpose(2, 3).contiguous().view(tasks_per_batch*n_way, -1, k_shot)

        dist = []
        for cc in range(tasks_per_batch*n_way):
            batch_idx = cc//n_way
            qq = query[batch_idx]
            uu, _, _ = torch.svd(class_representatives[cc].double())
            uu = uu.float()
            subspace = uu[:, :k_shot-1].transpose(0, 1)
            projection = subspace.transpose(0, 1).mm(subspace.mm(qq.transpose(0, 1))).transpose(0, 1)
            dist_perclass = torch.sum((qq - projection)**2, dim=-1)
            dist.append(dist_perclass)

        dist = torch.stack(dist).view(tasks_per_batch, n_way, -1).transpose(1, 2).contiguous().view(-1, n_way)
        logits = -dist / d

        query_y = query_y.view(tasks_per_batch * n_query)
        if self.training:
            loss = self.criterion(logits, query_y)
            return {"dsn_loss": loss}
        else:
            _, predict_labels = torch.max(logits, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
