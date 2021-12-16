try:
    import cv2
except ImportError:
    print("Can't find Opencv2, DeepEMD won't work properly")
    print("Hint: apt get update; apt-get install ffmpeg libsm6 libxext6 -y; pip install opencv-python")
    pass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from modules.utils import _l2norm
from .similarity import Similarity

@registry.Query.register("DeepEMD")
class DeepEMD(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()

        """
        @InProceedings{Zhang_2020_CVPR,
            author = {Zhang, Chi and Cai, Yujun and Lin, Guosheng and Shen, Chunhua},
            title = {DeepEMD: Few-Shot Image Classification With Differentiable Earth Mover's Distance and Structured Classifiers},
            booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
            month = {June},
            year = {2020}
        }

        https://github.com/icoz69/DeepEMD
        """

        self.cfg = cfg

        self.inner_simi = Similarity(cfg, metric='cosine')
        self.criterion = nn.CrossEntropyLoss()

        self.temperature = 12.5

    def emd_inference_opencv(self, cost_matrix, weight1, weight2):
        # cost matrix is a tensor of shape [N,N]
        cost_matrix = cost_matrix.detach().cpu().numpy()

        weight1 = F.relu(weight1) + 1e-5
        weight2 = F.relu(weight2) + 1e-5

        weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
        weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

        cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
        return cost, flow

    def get_weight_vector(self, A, B):

        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def normalize_feature(self, x):
        x = x - x.mean(1).unsqueeze(1)
        return x

    def get_emd_distance(self, similarity_map, weight_1, weight_2):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        for i in range(num_query):
            for j in range(num_proto):
                _, flow = self.emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

        temperature=(self.temperature / num_node)
        logitis = similarity_map.sum(-1).sum(-1) *  temperature
        return logitis

    def get_sfc(self, support_xf, support_y, n_way, k_shot):
        sfc_bs = 4
        
        b, s, c, h, w = support_xf.shape
        sfc = support_xf.view(b, n_way, k_shot, c, h, w).mean(2).clone().detach()
        sfc = nn.Parameter(sfc, requires_grad=True) # b, n, c, h, w
        
        optimizer = torch.optim.SGD([sfc], lr=100, momentum=0.9, dampening=0.9, weight_decay=0)
        label_shot = support_y.view(b, -1)
        label_shot = label_shot.long()

        with torch.enable_grad():
            for k in range(0, 20):
                rand_id = torch.randperm(n_way * k_shot).cuda()
                for j in range(0, n_way * k_shot, sfc_bs):
                    selected_id = rand_id[j: min(j + sfc_bs, n_way * k_shot)]
                    batch_shot = support_xf[:, selected_id, :, :, :]
                    batch_label = label_shot[:, selected_id].view(-1)
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(sfc, batch_shot.detach(), n_way)
                    loss = self.criterion(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return sfc
        
    def emd_forward_1shot(self, support_xf, query_xf, n_way):
        b, s, c, h, w = support_xf.shape
        q = query_xf.shape[1]

        logits = torch.zeros(b, q, n_way).to(query_xf.device)
        for i in range(b):
            query = query_xf[i] # [q, c, h, w]
            proto = support_xf[i] # [s, c, h, w]

            weight_1 = self.get_weight_vector(query, proto)
            weight_2 = self.get_weight_vector(proto, query)

            proto = self.normalize_feature(proto)
            query = self.normalize_feature(query)

            S = self.inner_simi(proto.view(-1, c, h * w).unsqueeze(0), query[None]).squeeze(0)
            logits[i] = self.get_emd_distance(S, weight_1, weight_2)
        logits = logits.view(b * q, n_way)
        return logits

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        b, s, c, h, w = support_xf.shape
        q = query_xf.shape[1]
        if self.training or k_shot == 1:
            support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w)[:, :, 0, :, :, :]
        else:
            support_xf = self.get_sfc(support_xf, support_y, n_way, k_shot)
        
        logits = self.emd_forward_1shot(support_xf, query_xf, n_way)
        query_y = query_y.view(-1)
        if self.training:
            loss = self.criterion(logits, query_y)
            return {"DeepEMD_loss": loss}
        else:
            _, predict_labels = torch.max(logits, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
