import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from modules.encoder import make_encoder
from modules.query.similarity import Similarity

class BasePretrainer(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        
        self.n_way = cfg.val.n_way
        self.k_shot = cfg.val.k_shot
        self.encoder = make_encoder(cfg)
        self.inner_simi = Similarity(cfg, metric='cosine')
        self.criterion = nn.CrossEntropyLoss()

    def forward_train(self, x, y):
        pass

    def forward_test(self, support_x, support_y, query_x, query_y):
        pass

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)
