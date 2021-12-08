import os.path as osp
from .collections import AttrDict

cfg = AttrDict()

cfg.seed = 1

# Default is DN4
cfg.model = AttrDict()
cfg.model.encoder = "FourLayer_64F"
cfg.model.forward_encoding = "FCN"
cfg.model.query = "DN4"

cfg.model.dn4 = AttrDict()
cfg.model.dn4.larger_shot = "group"

cfg.model.mn4 = AttrDict()
cfg.model.mn4.temperature = 2.0
cfg.model.mn4.larger_shot = "group"
cfg.model.mn4.is_norm = False

cfg.model.dmn4 = AttrDict()
cfg.model.dmn4.temperature = 2.0
cfg.model.dmn4.larger_shot = "group"
cfg.model.dmn4.is_norm = False

cfg.model.mel = AttrDict()
cfg.model.mel.k_q2s = -1
cfg.model.mel.k_s2q = -1
cfg.model.mel.gamma = 20.0
cfg.model.mel.gamma2 = 10.0
cfg.model.mel.katz_factor = 0.5

cfg.model.ls = AttrDict()
cfg.model.ls.gamma = 20.0

cfg.model.protonet = AttrDict()
cfg.model.protonet.temperature = 64.0
cfg.model.protonet.mel_mask = "query" # apply to query by default
cfg.model.protonet.mel_gamma = 20.0
cfg.model.protonet.mel_gamma2 = 10.0

cfg.model.relationnet = AttrDict()
cfg.model.relationnet.mel_mask = "query"
cfg.model.relationnet.mel_gamma = 20.0
cfg.model.relationnet.mel_gamma2 = 10.0

cfg.model.matchingnet = AttrDict()
cfg.model.matchingnet.temperature = 32.0
cfg.model.matchingnet.mel_mask = "query"
cfg.model.matchingnet.mel_gamma = 20.0
cfg.model.matchingnet.mel_gamma2 = 10.0

cfg.train = AttrDict()
cfg.train.query_per_class_per_episode = 15
cfg.train.episode_per_epoch = 20000
cfg.train.epochs = 30
cfg.train.colorjitter = False
cfg.train.learning_rate = 0.001
cfg.train.lr_decay = 0.1 
cfg.train.lr_decay_epoch = 10
cfg.train.lr_decay_milestones = []
cfg.train.lr_scheduler = "StepLR"
cfg.train.adam_betas = (0.5, 0.9)
cfg.train.sgd_mom = 0.9
cfg.train.optim = "Adam"
cfg.train.sgd_weight_decay = 5e-4
cfg.train.batch_size = 4
cfg.train.fix_bn = False
cfg.train.summary_snapshot_base = "./summary/"
cfg.train.n_way = 5
cfg.train.k_shot = 1
cfg.train.checkpoint_interval = -1
cfg.train.save_train_datalist = False
cfg.train.episode_first_dataloader = True

cfg.val = AttrDict()
cfg.val.episode = 1000
cfg.val.n_way = 5
cfg.val.k_shot = 1
cfg.val.query_per_class_per_episode = 15
cfg.val.epoch_start_val = 0
cfg.val.interval = -1

cfg.test = AttrDict()
cfg.test.query_per_class_per_episode = 15
cfg.test.episode = 10000
cfg.test.total_testtimes = 1
cfg.test.batch_size = 4
cfg.test.n_way = 5
cfg.test.k_shot = 1

cfg.data = AttrDict()
cfg.data.root = "./data/"
cfg.data.image_dir = "./data/miniImagenet"
cfg.data.image_size = 84
cfg.data.pad_size = 8

cfg.pre = AttrDict()
cfg.pre.pretrainer = "Linear"
cfg.pre.resolution = 25
cfg.pre.lr = 0.1
cfg.pre.lr_decay = 0.1
cfg.pre.lr_decay_milestones = [100, 200, 250, 300]
cfg.pre.lr_scheduler = "MultiStepLR"
cfg.pre.warmup_scheduler_epoch = -1
cfg.pre.snapshot_epoch = 200
cfg.pre.snapshot_interval = 5

cfg.pre.epochs = 350
cfg.pre.batch_size = 128
cfg.pre.colorjitter = True
cfg.pre.val_episode = 200
cfg.pre.pretrain_num_class = 64
