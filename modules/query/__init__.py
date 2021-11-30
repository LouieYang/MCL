from modules.query.linear import *
from modules.query.linearmel import *
from modules.query.linearplus import *
from modules.query.linearplusmel import *
from modules.query.dn4 import *
from modules.query.mn4 import *
from modules.query.dmn4 import *
from modules.query.relationnet import *
from modules.query.relationmel import *
from modules.query.protonet import *
from modules.query.protomel import *
from modules.query.matchingnet import *
from modules.query.matchingmel import *
from modules.query.mel import *
from modules.query.ls import *
from modules.query.frn import *
from modules.query.deepemd import *
from modules.query.dsn import *
from modules.query.dsnmel import *
from modules.query.ctx import *
from modules.query.negative_cosine import *
from modules.query.negative_cosine_mel import *
from modules.query.negative_softmax import *
from modules.query.negative_softmax_mel import *
from modules.query.metaoptnet import *
from modules.query.metaoptmel import *
from modules.query.r2d2 import *
from modules.query.r2d2mel import *

import modules.registry as registry

def make_query(in_channels, cfg):
    return registry.Query[cfg.model.query](in_channels, cfg)
