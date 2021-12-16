import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.registry as registry
from .base import BasePretrainer

@registry.Pretrainer.register("FRN")
class PretrainFRN(BasePretrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        resolution = cfg.pre.resolution
        self.category_mat = nn.Parameter(
            torch.randn(cfg.pre.pretrain_num_class, resolution, self.encoder.out_channels),
            requires_grad=True
        )
        self.r = nn.Parameter(torch.zeros(2),requires_grad=False)
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        self.num_category = cfg.pre.pretrain_num_class
        self.criterion = nn.NLLLoss()

    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)
        
        # correspond to lambda in the paper
        lam = reg*alpha.exp()+1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper
            
            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d
        
        else:
            # correspond to Equation 8 in the paper
            
            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf 
            hat = st.matmul(m_inv).matmul(support) # way, d, d

        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        return dist

    def get_neg_l2_dist(self, support_xf, query_xf):
        b, q, c, h, w = query_xf.shape

        support_xf = support_xf / math.sqrt(self.encoder.out_channels)
        query_xf = query_xf / math.sqrt(self.encoder.out_channels)

        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, -1).permute(0, 1, 2, 4, 3).contiguous()
        support_xf = support_xf.view(b, self.n_way, -1, c)
        query_xf = query_xf.view(b, q, c, -1).permute(0, 1, 3, 2).contiguous()
        query_xf = query_xf.view(b, q * h * w, c)

        alpha = self.r[0]
        beta = self.r[1]

        recon_dist = torch.zeros(b, q * h * w, self.n_way).to(query_xf.device)
        for i in range(b):
            recon_dist[i] = self.get_recon_dist(query_xf[i], support_xf[i], alpha, beta)
        neg_l2_dist = recon_dist.neg().view(-1, h * w, self.n_way).mean(1)
        return neg_l2_dist

    def forward_train(self, x, y):
        enc = self.encoder(x)
        enc = enc / math.sqrt(self.encoder.out_channels)
        b, c, h, w = enc.shape
        enc = enc.view(b, c, h * w).permute(0, 2, 1).contiguous()
        enc = enc.view(b * h * w, c)
        alpha, beta = self.r[0], self.r[1]
        recon_dist = self.get_recon_dist(query=enc,support=self.category_mat,alpha=alpha,beta=beta)
        neg_l2_dist = recon_dist.neg().view(b, h * w, self.num_category).mean(1)
        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits,dim=1)
        loss = self.criterion(log_prediction, y)
        return {"pretrain_frn_loss": loss}

    def forward_test(self, support_x, support_y, query_x, query_y):
        b, q = query_x.shape[:2]
        s = support_x.shape[1]

        support_x = support_x.view((-1,) + support_x.shape[-3:])
        support_x = self.encoder(support_x)
        support_x = support_x.view((b, s) + support_x.shape[-3:])
        query_x = query_x.view((-1,) + query_x.shape[-3:])
        query_x = self.encoder(query_x)
        query_x = query_x.view((b, q) + query_x.shape[-3:])

        neg_l2_dist = self.get_neg_l2_dist(support_x, query_x)
        logits = neg_l2_dist * self.scale
        query_y = query_y.view(-1)
        _, predict_labels = torch.max(neg_l2_dist, 1)
        rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
        return rewards
