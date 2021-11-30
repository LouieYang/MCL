import os
import os.path as osp
import sys
import argparse

import shutil

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from configs.miniimagenet_default import cfg
from engines.evaluator import evaluator as e
from experiments.utils import cfg_to_dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument("-d",'--device', type=str, dest='device', default='0')
    parser.add_argument('-c', '--checkpoint', type=str, default='')
    parser.add_argument('-b', '--checkpoint_base', type=str, default='./checkpoint')
    parser.add_argument('-s', '--snapshot_base', type=str, dest='snapshot_base', default='./snapshots')
    parser.add_argument('-e', '--eval_store', type=int, dest='eval_store', default=1)
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    dataset = cfg_to_dataset(args.cfg)
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.rest:
        cfg.merge_from_list(args.rest)

    snapshot_dir = osp.join(args.snapshot_base, dataset + "_" + osp.basename(args.cfg).replace(".yaml", ""))
    if not args.checkpoint:
        args.checkpoint = dataset + "_" + osp.basename(args.cfg).replace(".yaml", "")
        args.checkpoint = osp.join(args.checkpoint_base, args.checkpoint, "ebest_{}way_{}shot.pth".format(cfg.train.n_way, cfg.train.k_shot))
        if not osp.exists(args.checkpoint):
            raise FileNotFoundError

    cfg.data.image_dir = osp.join(cfg.data.root, dataset)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    print("[*] Source Image Path: {}".format(cfg.data.image_dir))
    print("[*] Source Checkpoint Path: {}".format(args.checkpoint))

    evaluator = e(cfg, args.checkpoint)
    accuracy = evaluator.run()
    if args.eval_store:
        shutil.copyfile(args.cfg, osp.join(snapshot_dir, osp.basename(args.cfg)))
        shutil.copyfile(args.checkpoint, osp.join(snapshot_dir, osp.basename(args.checkpoint)))
        shutil.copyfile(args.checkpoint.replace(".pth", ".txt"), osp.join(snapshot_dir, osp.basename(args.checkpoint.replace(".pth", ".txt"))))
        shutil.copyfile(evaluator.prediction_dir, osp.join(snapshot_dir, osp.basename(evaluator.prediction_dir)))

        target_snapshot_path = snapshot_dir + "_{:.3f}".format(accuracy * 100)
        shutil.move(snapshot_dir, target_snapshot_path)

if __name__ == "__main__":
    main()

