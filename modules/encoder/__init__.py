from modules.encoder.fourlayer_64F import *
from modules.encoder.fourlayer_64F_4x import *
from modules.encoder.resnet import *
from modules.encoder.wrn28 import *

import modules.registry as registry

def make_encoder(cfg):
    return registry.Encoder[cfg.model.encoder]()
