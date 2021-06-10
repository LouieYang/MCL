import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import make_encoder
from .query.innerproduct_similarity import InnerproductSimilarity

class PretrainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.encoder = make_encoder(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.fake_classifier = nn.Linear(self.encoder.out_channels, cfg.pre.pretrain_num_class)
        self.avg_pool = nn.AvgPool2d(5, stride=1)

        self.inner_simi = InnerproductSimilarity(cfg, metric='cosine')
        self.mel_gamma = cfg.model.mel.gamma
        self.eps_ = 1e-6

    def forward(self, x, y):
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

    def forward_dn4(self, support_x, support_y, query_x, query_y):
        b, q = query_x.shape[:2]
        s = support_x.shape[1]
        
        support_x = support_x.view((-1,) + support_x.shape[-3:])
        support_x = self.encoder(support_x)
        support_x = support_x.view((b, s) + support_x.shape[-3:])
        query_x = query_x.view((-1,) + query_x.shape[-3:])
        query_x = self.encoder(query_x)
        query_x = query_x.view((b, q) + query_x.shape[-3:])

        innerproduct_matrix = self.inner_simi(support_x, support_y, query_x, query_y)
        topk_value, _ = torch.topk(innerproduct_matrix, 1, -1) # [b, q, N, M_q, neighbor_k]
        similarity_matrix = topk_value.mean(-1).view(b, q, self.n_way, -1).sum(-1)
        similarity_matrix = similarity_matrix.view(b * q, self.n_way)

        query_y = query_y.view(b * q)
        _, predict_labels = torch.max(similarity_matrix, 1)
        rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
        return rewards

    def forward_proto(self, support_x, support_y, query_x, query_y):
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

    def forward_mel(self, support_x, support_y, query_x, query_y):
        katz_factor = 0.999
        b, q = query_x.shape[:2]
        s = support_x.shape[1]
        
        support_x = support_x.view((-1,) + support_x.shape[-3:])
        support_x = self.encoder(support_x)
        support_xf = support_x.view((b, s) + support_x.shape[-3:])
        query_x = query_x.view((-1,) + query_x.shape[-3:])
        query_x = self.encoder(query_x)
        query_xf = query_x.view((b, q) + query_x.shape[-3:])

        device = support_xf.device

        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        S = self.inner_simi(support_xf, support_y, query_xf, query_y) # [b, q, N, M_q, M_s]
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        N_examples, M_q, M_s = S.shape
        St = S.transpose(-2, -1)
        device = S.device

        T_sq = torch.exp(self.mel_gamma * (S - S.max(-1, keepdim=True)[0]))
        T_sq = T_sq / T_sq.sum(-1, keepdim=True) # row-wise stochastic
        T_qs = torch.exp(self.mel_gamma * (St - St.max(-1, keepdim=True)[0])) # [b * q, M_s, M_q]
        T_qs = T_qs / T_qs.sum(-1, keepdim=True) # row-wise stochastic

        T = torch.cat([
            torch.cat([torch.zeros((N_examples, M_s, M_s), device=device), T_sq.transpose(-2, -1)], dim=-1),
            torch.cat([T_qs.transpose(-2, -1), torch.zeros((N_examples, M_q, M_q), device=device)], dim=-1),
        ], dim=-2)
        katz = (torch.inverse(torch.eye(M_s + M_q, device=device)[None].repeat(N_examples, 1, 1) - katz_factor * T) - \
                torch.eye(M_s + M_q, device=S.device)[None].repeat(N_examples, 1, 1))@torch.ones((N_examples, M_s + M_q, 1), device=device)
        partial_katz = katz.squeeze(-1)[:, :M_s] / katz.squeeze(-1)[:, :M_s].sum(-1, keepdim=True)
        predicts = partial_katz.view(N_examples, self.n_way, -1).sum(-1)

        query_y = query_y.view(N_examples)
        _, predict_labels = torch.max(predicts, 1)
        rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
        return rewards

def make_pretrain_model(cfg):
    return PretrainModel(cfg)
