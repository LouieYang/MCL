from modules.query.linear import *
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

import modules.registry as registry

def make_query(in_channels, cfg):
    return registry.Query[cfg.model.query](in_channels, cfg)
