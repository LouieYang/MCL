def cfg_to_dataset(cfg):
    dataset = ["miniImagenet", "CUB_FSL", "tieredimagenet", "Aircraft_fewshot", "tiered_meta_iNat", "meta_iNat"]
    for d in dataset:
        if d in cfg:
            return d
    raise FileNotFoundError("{} doesn't include any dataset information".format(cfg))
