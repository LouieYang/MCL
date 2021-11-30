import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from torch.nn.utils.weight_norm import WeightNorm

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

@registry.Query.register("Linear++")
class LinearPlus(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @inproceedings{chen2019closerfewshot,
            title={A Closer Look at Few-shot Classification},
            author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang and  Huang, Jia-Bin},
            booktitle={International Conference on Learning Representations},
            year={2019}
        }

        https://github.com/wyharveychen/CloserLookFewShot
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
        
        linear_clf = distLinear(self.in_channels, self.n_way)
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
                    scores = linear_clf(s_batch)
                    loss = self.criterion(scores, y_batch)
                    loss.backward()
                    linear_optim.step()
        scores = linear_clf(query_xf)

        _, predict_labels = torch.max(scores, 1)
        query_y = query_y.view(-1)
        rewards = [1 if predict_labels[j]==query_y[j].to(device) else 0 for j in range(len(query_y))]
        return rewards
