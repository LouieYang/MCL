import os
import os.path as osp
import sys
import argparse

import shutil

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from configs.miniimagenet_default import cfg
from engines.trainer import trainer as t
from experiments.utils import cfg_to_dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument("-d",'--device', type=str, dest='device', default='0')
    parser.add_argument('-c', '--checkpoint_base', type=str, dest='checkpoint_base', default='./checkpoint')
    parser.add_argument('-s', '--snapshot_base', type=str, dest='snapshot_base', default='./snapshots')
    parser.add_argument('-e', '--eval_after_train', type=int, dest='eval_after_train', default=1)
    parser.add_argument('-p', '--pretrain_path', type=str, default='')
    parser.add_argument('--prefix', type=str, default='')
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

    if args.pretrain_path and osp.isfile(args.pretrain_path):
        link_pretrain = osp.join(checkpoint_dir, osp.basename(args.pretrain_path))
        if not osp.exists(link_pretrain):
            os.symlink(osp.abspath(args.pretrain_path), link_pretrain)

    print("[*] Source Image Path: {}".format(cfg.data.image_dir))
    print("[*] Target Checkpoint Path: {}".format(checkpoint_dir))

    trainer = t(cfg, checkpoint_dir)
    trainer.run()

    shutil.copyfile(args.cfg, osp.join(snapshot_dir, osp.basename(args.cfg)))
    shutil.copyfile(trainer.snapshot_name("best"), osp.join(snapshot_dir, osp.basename(trainer.snapshot_name("best"))))
    shutil.copyfile(trainer.snapshot_record("best"), osp.join(snapshot_dir, osp.basename(trainer.snapshot_record("best"))))
    shutil.copytree(trainer.writer_dir, osp.join(snapshot_dir, osp.basename(trainer.writer_dir)))

    if args.eval_after_train:
        print("[*] Running Evaluations ...")
        from engines.evaluator import evaluator as e
        evaluator = e(cfg, trainer.snapshot_name("best"))
        accuracy = evaluator.run()
        shutil.copyfile(evaluator.prediction_dir, osp.join(snapshot_dir, osp.basename(evaluator.prediction_dir)))
        target_snapshot_path = snapshot_dir + "_{:.3f}".format(accuracy * 100)
        shutil.move(snapshot_dir, target_snapshot_path)

        print("[*] Saving to {}".format(osp.abspath(target_snapshot_path)))

if __name__ == "__main__":
    main()

