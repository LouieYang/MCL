import os
import os.path as osp
import sys
import argparse
import shutil

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from configs.miniimagenet_default import cfg
from engines.pretrainer import Pretrainer as t
from experiments.utils import cfg_to_dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument("-d",'--device', type=str, dest='device', default='0')
    parser.add_argument('-p', '--prefix', type=str, default='')
    parser.add_argument('-c', '--checkpoint_base', type=str, dest='checkpoint_base', default='./checkpoint')
    parser.add_argument('-s', '--snapshot_base', type=str, dest='snapshot_base', default='./snapshots')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    dataset = cfg_to_dataset(args.cfg)
    if not args.prefix:
        args.prefix = dataset + "_" + osp.basename(args.cfg).replace(".yaml", "")

    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.rest:
        cfg.merge_from_list(args.rest)
    cfg.data.image_dir = osp.join(cfg.data.root, dataset)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    checkpoint_dir = osp.join(args.checkpoint_base, args.prefix)
    snapshot_dir = osp.join(args.snapshot_base, args.prefix)
    for d in [checkpoint_dir, snapshot_dir]:
        if not osp.exists(d):
            os.mkdir(d)
    print("[*] Source Image Path: {}".format(cfg.data.image_dir))
    print("[*] Target Checkpoint Path: {}".format(checkpoint_dir))

    trainer = t(cfg, checkpoint_dir)
    trainer.run()

    shutil.copyfile(args.cfg, osp.join(snapshot_dir, osp.basename(args.cfg)))
    shutil.copyfile(trainer.snapshot_for_meta, osp.join(snapshot_dir, osp.basename(trainer.snapshot_for_meta)))
    shutil.copytree(trainer.writer_dir, osp.join(snapshot_dir, osp.basename(trainer.writer_dir)))
if __name__ == "__main__":
    main()

