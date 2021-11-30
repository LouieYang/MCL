import os
import os.path as osp
import sys
import argparse

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from configs.miniimagenet_default import cfg
from engines.pretrainer import Pretrainer as t
from experiments.utils import cfg_to_dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument("-d",'--device', type=str, dest='device', default='0')
    parser.add_argument('-p', '--checkpoint_dir', type=str, default='')
    parser.add_argument('-c', '--checkpoint_base', type=str, dest='checkpoint_base', default='./checkpoint')
    args = parser.parse_args()

    dataset = cfg_to_dataset(args.cfg)
    if not args.checkpoint_dir:
        args.checkpoint_dir = dataset + "_" + osp.basename(args.cfg).replace(".yaml", "")

    if args.cfg:
        cfg.merge_from_file(args.cfg)
    cfg.data.image_dir = osp.join(cfg.data.root, dataset)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    checkpoint_dir = osp.join(args.checkpoint_base, args.checkpoint_dir)
    if not osp.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    print("[*] Source Image Path: {}".format(cfg.data.image_dir))
    print("[*] Target Checkpoint Path: {}".format(checkpoint_dir))

    trainer = t(cfg, checkpoint_dir)
    trainer.run()

if __name__ == "__main__":
    main()

