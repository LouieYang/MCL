import os
import os.path as osp
import sys
import argparse

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from configs.miniimagenet_default import cfg
from evaluator import evaluator as e

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', default='')
    parser.add_argument("-d",'--device', type=str, dest='device', default='0')
    parser.add_argument('-c', '--checkpoint', type=str, default='')
    parser.add_argument('-b', '--checkpoint_base', type=str, default='./checkpoint')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    pardir = osp.basename(osp.abspath(osp.join(args.cfg, osp.pardir)))
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.rest:
        cfg.merge_from_list(args.rest)

    if not args.checkpoint:
        args.checkpoint = pardir + "_" + osp.basename(args.cfg).replace(".yaml", "")
        args.checkpoint = osp.join(args.checkpoint_base, args.checkpoint, "ebest_{}way_{}shot.pth".format(cfg.n_way, cfg.k_shot))
        if not osp.exists(args.checkpoint):
            raise FileNotFoundError
    cfg.data.image_dir = osp.join(cfg.data.root, pardir)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    print("[*] Source Image Path: {}".format(cfg.data.image_dir))
    print("[*] Source Checkpoint Path: {}".format(args.checkpoint))

    evaluator = e(cfg, args.checkpoint)
    evaluator.run()

if __name__ == "__main__":
    main()

