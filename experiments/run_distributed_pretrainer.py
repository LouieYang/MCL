import os
import os.path as osp
import sys
import argparse
import torch
import shutil

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from configs.miniimagenet_default import cfg
from engines.distributed_pretrainer import DistributedPretrainer as t
from engines.distributed_utils import synchronize, is_main_process
from experiments.utils import cfg_to_dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument('-p', '--checkpoint_dir', type=str, default='')
    parser.add_argument('-c', '--checkpoint_base', type=str, dest='checkpoint_base', default='./checkpoint')
    parser.add_argument('--local_rank', '-r', type=int, default=0)
    parser.add_argument('--base_rank', '-b', type=int, default=0)
    parser.add_argument('--world-size', '-w', type=int, default=1)
    parser.add_argument('--init-method', '-i', type=str, default='env://')
    parser.add_argument('--socket-ifname', '-s', type=str, default='lo')

    args = parser.parse_args()

    os.environ["NCCL_SOCKET_IFNAME"] = args.socket_ifname
    dataset = cfg_to_dataset(args.cfg)
    if not args.checkpoint_dir:
        args.checkpoint_dir = dataset + "_" + osp.basename(args.cfg).replace(".yaml", "")
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    cfg.data.image_dir = osp.join(cfg.data.root, dataset)

    args.distributed = args.world_size > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method=args.init_method, 
            rank=args.local_rank + args.base_rank, 
            world_size=args.world_size
        )
        synchronize()

    checkpoint_dir = osp.join(args.checkpoint_base, args.checkpoint_dir)
    if is_main_process():
        for d in [checkpoint_dir]:
            if not osp.exists(d):
                os.mkdir(d)
        print("[*] Target Checkpoint Path: {}".format(checkpoint_dir))

    trainer = t(args, cfg, checkpoint_dir)
    trainer.run()

if __name__ == "__main__":
    main()

