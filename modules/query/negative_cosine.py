import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    :return： (theta) - m
    """

    def __init__(self, in_features, out_features, scale_factor=30.0, margin=0.40):
        super(AddMarginProduct, self).__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))

        # when test, no label, just return
        if label is None:
            return cosine * self.scale_factor

        phi = cosine - self.margin
        output = torch.where(
            torch.nn.functional.one_hot(label, cosine.shape[1]).byte(), phi, cosine)
        output *= self.scale_factor

        return output

@registry.Query.register("NegativeCosine")
class NegativeCosine(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @inproceedings{liu2020negative,
            title={Negative margin matters: Understanding margin in few-shot classification},
            author={Liu, Bin and Cao, Yue and Lin, Yutong and Li, Qi and Zhang, Zheng and Long, Mingsheng and Hu, Han},
            booktitle={European Conference on Computer Vision},
            pages={438--455},
            year={2020},
            organization={Springer}
        }

        https://github.com/bl0/negative-margin.few-shot
        """

        self.n_way = cfg.test.n_way
        self.k_shot = cfg.test.k_shot

        self.in_channels = in_channels
        self.criterion = nn.CrossEntropyLoss()

    def pooling(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        b, q, c, h, w = query_xf.shape
        assert b == 1
        query_xf = F.adaptive_avg_pool2d(query_xf.squeeze(0), 1).squeeze(-1).squeeze(-1) # [q, c]
        query_xf = query_xf.clone().detach()

        b, s, c, h, w = support_xf.shape
        assert s == self.n_way * self.k_shot
        assert b == 1
        support_xf = F.adaptive_avg_pool2d(support_xf.squeeze(0), 1).squeeze(-1).squeeze(-1) # [s, c]
        support_xf = support_xf.clone().detach()
        support_y = support_y.view(s)
        return support_xf, support_y, query_xf, query_y

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot
        assert not self.training
        device = support_xf.device
        
        linear_clf = AddMarginProduct(self.in_channels, self.n_way)
        linear_clf = linear_clf.to(device)

        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]
        support_xf, support_y, query_xf, query_y = self.pooling(support_xf, support_y, query_xf, query_y, n_way, k_shot)

        linear_optim = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        batch_size = 4
        with torch.enable_grad():
            for epoch in range(100):
                rand_id = np.random.permutation(s)
                for i in range(0, s, batch_size):
                    linear_optim.zero_grad()
                    selected_id = torch.from_numpy(rand_id[i: min(i+batch_size, s)]).to(device)
                    s_batch = support_xf[selected_id]
                    y_batch = support_y[selected_id]
                    scores = linear_clf(s_batch, y_batch)
                    loss = self.criterion(scores, y_batch)
                    loss.backward()
                    linear_optim.step()
        scores = linear_clf(query_xf)

        _, predict_labels = torch.max(scores, 1)
        query_y = query_y.view(-1)
        rewards = [1 if predict_labels[j]==query_y[j].to(device) else 0 for j in range(len(query_y))]
        return rewards
