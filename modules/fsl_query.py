import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import make_encoder
from .query import make_query

class FSLQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = make_encoder(cfg)
        self.query = make_query(self.encoder.out_channels, cfg)

        self.forward_encoding = cfg.model.forward_encoding
        self.pyramid_list = self._parse_encoding_params()

        self.cfg = cfg

    def _parse_encoding_params(self):
        idx = self.forward_encoding.find('-')
        if idx < 0:
            return []
        blocks = self.forward_encoding[idx + 1:].split(',')
        blocks = [int(s) for s in blocks]
        return blocks

    def _pyramid_encoding(self, x):
        b, n, c, h , w = x.shape
        x = x.view(-1, c, h, w)
        feature_list = []
        for size_ in self.pyramid_list:
            feature_list.append(F.adaptive_avg_pool2d(x, size_).view(b, n, c, 1, -1))

        if not feature_list:
            out = x.view(b, n, c, 1, -1)
        else:
            out = torch.cat(feature_list, dim=-1)
        return out

    def forward_Grid(self, support_x, support_y, query_x, query_y):
        b, s, grids_sc, h, w = support_x.shape
        grids_s = grids_sc // 3
        _, q, grids_qc  = query_x.shape[:3]
        grids_q = grids_qc // 3
        
        support_xf = F.adaptive_avg_pool2d(self.encoder(support_x.view(-1, 3, h, w)), 1)
        support_xf = support_xf.view(b, s, grids_s, -1).permute(0, 1, 3, 2).unsqueeze(-1)
        query_xf = F.adaptive_avg_pool2d(self.encoder(query_x.view(-1, 3, h, w)), 1)
        query_xf = query_xf.view(b, q, grids_q, -1).permute(0, 1, 3, 2).unsqueeze(-1)

        return support_xf, query_xf

    def forward_PyramidFCN(self, support_x, support_y, query_x, query_y):
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]

        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        fc, fh, fw = support_xf.shape[-3:]
        support_xf = support_xf.view(b, s, fc, fh, fw)
        query_xf = query_xf.view(b, q, fc, fh, fw)

        support_xf = self._pyramid_encoding(support_xf)
        query_xf = self._pyramid_encoding(query_xf)
        return support_xf, query_xf

    def forward_FCN(self, support_x, support_y, query_x, query_y):
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]

        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        fc, fh, fw = support_xf.shape[-3:]
        support_xf = support_xf.view(b, s, fc, fh, fw)
        query_xf = query_xf.view(b, q, fc, fh, fw)

        return support_xf, query_xf

    def horizontal_flip_aug(self, support_x, support_y, k_shot):
        b, s, c, h, w = support_x.shape
        support_x = support_x.view(b, -1, k_shot, c, h, w)
        reflected_support_x = torch.flip(support_x, (-1,))
        support_x = torch.cat([support_x, reflected_support_x], dim=2).view(b, -1, c, h, w)
        support_y = support_y.view(b, -1, k_shot)
        support_y = torch.cat([support_y, support_y], dim=-1).view(b, -1)
        k_shot = 2 * k_shot
        return support_x, support_y, k_shot


    def forward(self, support_x, support_y, query_x, query_y, n_way, k_shot):
        if self.cfg.model.query.startswith("DSN") and k_shot == 1:
            support_x, support_y, k_shot = self.horizontal_flip_aug(support_x, support_y, k_shot)

        if self.forward_encoding == "FCN":
            support_xf, query_xf = self.forward_FCN(support_x, support_y, query_x, query_y)
        elif self.forward_encoding.startswith("Grid"):
            support_xf, query_xf = self.forward_Grid(support_x, support_y, query_x, query_y)
        elif self.forward_encoding.startswith("PyramidFCN"):
            support_xf, query_xf = self.forward_PyramidFCN(support_x, support_y, query_x, query_y)
        else:
            raise NotImplementedError
        query = self.query(support_xf, support_y, query_xf, query_y, n_way, k_shot)

        if self.training:
            query = sum(query.values())
        return query

def make_fsl(cfg):
    return FSLQuery(cfg)
