import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

@registry.Query.register("FRN")
class FRN(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @InProceedings{Wertheimer_2021_CVPR,
            author    = {Wertheimer, Davis and Tang, Luming and Hariharan, Bharath},
            title     = {Few-Shot Classification With Feature Map Reconstruction Networks},
            booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
            month     = {June},
            year      = {2021},
            pages     = {8012-8021}
        }

        https://github.com/Tsingularity/FRN
        """

        self.cfg = cfg

        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.r = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.d = in_channels

        self.criterion = nn.CrossEntropyLoss()

    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)

        # correspond to lambda in the paper
        lam = reg*alpha.exp()

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

        support_xf = support_xf / math.sqrt(self.d)
        query_xf = query_xf / math.sqrt(self.d)

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

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        neg_l2_dist = self.get_neg_l2_dist(support_xf, query_xf)
        logits = neg_l2_dist * self.scale
        query_y = query_y.view(-1)
        if self.training:
            loss = self.criterion(logits, query_y)
            return {"FRN_loss": loss}
        else:
            _, predict_labels = torch.max(neg_l2_dist, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
