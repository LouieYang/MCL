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
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('-c', '--checkpoint_base', type=str, dest='checkpoint_base', default='./checkpoint')
    parser.add_argument('--local_rank', '-r', type=int, default=0)
    parser.add_argument('--base_rank', '-b', type=int, default=0)
    parser.add_argument('--init-method', '-i', type=str, default='env://')
    parser.add_argument('--socket-ifname', '-s', type=str, default='lo')

    parser.add_argument('-s', '--snapshot_base', type=str, dest='snapshot_base', default='./snapshots')
    args = parser.parse_args()

    os.environ["NCCL_SOCKET_IFNAME"] = args.socket_ifname
    dataset = cfg_to_dataset(args.cfg)
    if not args.prefix:
        args.prefix = dataset + "_" + osp.basename(args.cfg).replace(".yaml", "")
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    cfg.data.image_dir = osp.join(cfg.data.root, dataset)

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = True
    else:
        return

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method,
        rank=args.local_rank + args.base_rank,
        world_size=world_size
    )
    synchronize()

    checkpoint_dir = osp.join(args.checkpoint_base, args.prefix)
    snapshot_dir = osp.join(args.snapshot_base, args.prefix)
    if is_main_process():
        for d in [checkpoint_dir, snapshot_dir]:
            if not osp.exists(d):
                os.mkdir(d)
        print("[*] Target Checkpoint Path: {}".format(checkpoint_dir))

    trainer = t(args, cfg, checkpoint_dir)
    trainer.run()

    if is_main_process():
        shutil.copyfile(args.cfg, osp.join(snapshot_dir, osp.basename(args.cfg)))
        shutil.copyfile(trainer.snapshot_for_meta, osp.join(snapshot_dir, osp.basename(trainer.snapshot_for_meta)))
        shutil.copytree(trainer.writer_dir, osp.join(snapshot_dir, osp.basename(trainer.writer_dir)))

if __name__ == "__main__":
    main()

