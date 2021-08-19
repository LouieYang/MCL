from .linear import *

import modules.registry as registry

def make_pretrain_model(cfg):
    return registry.Pretrainer[cfg.pre.pretrainer](cfg)
