from .linear import *
from .frn import *
from .dn4 import *
from .mel import *

import modules.registry as registry

def make_pretrain_model(cfg):
    return registry.Pretrainer[cfg.pre.pretrainer](cfg)
