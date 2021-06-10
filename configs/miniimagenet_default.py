import os.path as osp
from .collections import AttrDict

cfg = AttrDict()

cfg.seed = 1

# Default is DN4
cfg.model = AttrDict()
cfg.model.encoder = "FourLayer_64F"
cfg.model.forward_encoding = "FCN"
cfg.model.query = "DN4"
cfg.model.nbnn_topk = 1

cfg.model.mel = AttrDict()
cfg.model.mel.k_q2s = -1
cfg.model.mel.k_s2q = -1
cfg.model.mel.gamma = 15.0
cfg.model.mel.gamma2 = 5.0
cfg.model.mel.katz_factor = 0.5

cfg.model.ls = AttrDict()
cfg.model.ls.gamma = 20.0

cfg.model.protonet = AttrDict()
cfg.model.protonet.temperature = 64
cfg.model.protonet.mel_mask = "query" # apply to query by default

cfg.model.relationnet = AttrDict()
cfg.model.relationnet.mel_mask = "query"

cfg.model.matchingnet = AttrDict()
cfg.model.matchingnet.temperature = 32
cfg.model.matchingnet.mel_mask = "query"

cfg.n_way = 5
cfg.k_shot = 5

cfg.train = AttrDict()
cfg.train.query_per_class_per_episode = 10
cfg.train.episode_per_epoch = 20000
cfg.train.epochs = 30
cfg.train.colorjitter = False
cfg.train.learning_rate = 0.001
cfg.train.lr_decay = 0.1 
cfg.train.lr_decay_epoch = 10
cfg.train.lr_decay_milestones = []
cfg.train.adam_betas = (0.5, 0.9)
cfg.train.sgd_mom = 0.9
cfg.train.optim = "Adam"
cfg.train.sgd_weight_decay = 5e-4
cfg.train.batch_size = 4
cfg.train.fix_bn = False
cfg.train.summary_snapshot_base = "./summary/"

cfg.val = AttrDict()
cfg.val.episode = 1000

cfg.test = AttrDict()
cfg.test.query_per_class_per_episode = 15
cfg.test.episode = 10000
cfg.test.total_testtimes = 1
cfg.test.batch_size = 4

cfg.data = AttrDict()
cfg.data.root = "./data/"
cfg.data.image_dir = "./data/miniImagenet"

cfg.pre = AttrDict()
cfg.pre.lr = 0.1
cfg.pre.lr_decay = 0.1
cfg.pre.lr_decay_milestones = [100, 200, 250, 300]
cfg.pre.snapshot_epoch = 200
cfg.pre.snapshot_interval = 5

cfg.pre.epochs = 350
cfg.pre.batch_size = 128
cfg.pre.colorjitter = True
cfg.pre.val_episode = 200
cfg.pre.pretrain_num_class = 64
