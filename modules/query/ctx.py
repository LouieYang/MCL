import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

@registry.Query.register("CTX")
class CTX(nn.Module):

    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @article{doersch2020crosstransformers,
              title={Crosstransformers: spatially-aware few-shot transfer},
              author={Doersch, Carl and Gupta, Ankush and Zisserman, Andrew},
              journal={arXiv preprint arXiv:2007.11498},
              year={2020}
        }

        https://github.com/lucidrains/cross-transformers-pytorch
        """

        self.cfg = cfg
        D = in_channels
        Ds = in_channels
        dk, dv = 128, 128
        self.key_head = nn.Conv2d(Ds, dk, 1, bias=False)
        self.query_head = nn.Conv2d(D, dk, 1, bias=False)
        self.value_head = nn.Conv2d(Ds, dv, 1, bias=False)

        self.criterion = nn.CrossEntropyLoss()

    def forward_per_batch(self, query_image_features, support_image_features):
        """ 
        query B x D x H x W
        support Nc x Nk x Ds x Hs x Ws (#CLASSES x #SHOT x #DIMENSIONS)
        """

        Nc, Nk, Ds, Hs, Ws = support_image_features.shape
        support_image_features = support_image_features.view(Nc * Nk, Ds, Hs, Ws)

        ### Step 1: Get query and support features
        # query_image_features = self.feature_extractor(query)
        # support_image_features = self.feature_extractor(support.view(Nc*Nk, Ds, Hs, Ws))


        ### Step 2: Calculate query aligned prototype
        query = self.query_head(query_image_features)
        support_key = self.key_head(support_image_features)
        support_value = self.value_head(support_image_features)

        dk = query.shape[1]

        ## flatten pixels in query (p in the paper)
        query = query.view(query.shape[0], query.shape[1], -1)

        ## flatten pixels & k-shot in support (j & m in the paper respectively)
        support_key = support_key.view(Nc, Nk, support_key.shape[1], -1)
        support_value = support_value.view(Nc, Nk, support_value.shape[1], -1)

        support_key = support_key.permute(0, 2, 3, 1).contiguous()
        support_value = support_value.permute(0, 2, 3, 1).contiguous()

        support_key = support_key.view(Nc, support_key.shape[1], -1)
        support_value = support_value.view(Nc, support_value.shape[1], -1)

        ## v is j images' m pixels, ie k-shot*h*w
        attn_weights = torch.einsum('bdp,ndv->bnpv', query, support_key) * (dk ** -0.5)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        ## get weighted sum of support values
        support_value = support_value.unsqueeze(0).expand(attn_weights.shape[0], -1, -1, -1)
        query_aligned_prototype = torch.einsum('bnpv,bndv->bnpd', attn_weights, support_value)

        ### Step 3: Calculate query value
        query_value = self.value_head(query_image_features)
        query_value = query_value.view(query_value.shape[0], -1, query_value.shape[1]) ##bpd

        ### Step 4: Calculate distance between queries and supports
        distances = []
        for classid in range(query_aligned_prototype.shape[1]):
            dxc = torch.cdist(query_aligned_prototype[:, classid].contiguous(),
                                            query_value, p=2)
            dxc = dxc**2
            B,P,R = dxc.shape
            dxc = dxc.sum(dim=(1,2)) / (P*R)
            distances.append(dxc)

        distances = torch.stack(distances, dim=1)

        return distances

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        dist = []
        for i in range(b):
            dist.append(self.forward_per_batch(query_xf[i], support_xf[i].view(self.n_way, self.k_shot, c, h, w)))
        dist = torch.cat(dist, dim=0)
        logits = -dist
        query_y = query_y.view(-1)
        if self.training:
            loss = self.criterion(logits, query_y)
            return {"CTX_loss": loss}
        else:
            _, predict_labels = torch.max(logits, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
