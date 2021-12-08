import os
import os.path as osp
import sys
import argparse
import torch
import shutil

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from configs.miniimagenet_default import cfg
from engines.distributed_trainer import DistributedTrainer as t
from engines.distributed_utils import synchronize, is_main_process
from experiments.utils import cfg_to_dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument('-c', '--checkpoint_base', type=str, dest='checkpoint_base', default='./checkpoint')
    parser.add_argument('--local_rank', '-r', type=int, default=0)
    parser.add_argument('--base_rank', '-b', type=int, default=0)
    parser.add_argument('--init-method', '-i', type=str, default='env://')
    parser.add_argument('--socket-ifname', '-s', type=str, default='lo')

    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--snapshot_base', type=str, dest='snapshot_base', default='./snapshots')
    parser.add_argument('-p', '--pretrain_path', type=str, default='')
    parser.add_argument('-e', '--eval_after_train', type=int, dest='eval_after_train', default=1)
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
        if args.pretrain_path and osp.isfile(args.pretrain_path):
            link_pretrain = osp.join(checkpoint_dir, osp.basename(args.pretrain_path))
            if not osp.exists(link_pretrain):
                os.symlink(osp.abspath(args.pretrain_path), link_pretrain)
        print("[*] Target Checkpoint Path: {}".format(checkpoint_dir))

    trainer = t(args, cfg, checkpoint_dir)
    trainer.run()

    if is_main_process():
        shutil.copyfile(args.cfg, osp.join(snapshot_dir, osp.basename(args.cfg)))
        shutil.copyfile(trainer.snapshot_name("best"), osp.join(snapshot_dir, osp.basename(trainer.snapshot_name("best"))))
        shutil.copyfile(trainer.snapshot_record("best"), osp.join(snapshot_dir, osp.basename(trainer.snapshot_record("best"))))
        shutil.copytree(trainer.writer_dir, osp.join(snapshot_dir, osp.basename(trainer.writer_dir)))

    # if args.eval_after_train:
    #     fsl = trainer.fsl
    #     trainer.fsl.load_state_dict(trainer.best_state_dict_for_distributed)
    #     if is_main_process():
    #         print("[*] Running Evaluations ...")
    #     evaluator = e(args, cfg, trainer.snapshot_name("best"), fsl)
    #     accuracy = evaluator.run()
    #     if is_main_process():
    #         shutil.copyfile(evaluator.prediction_dir, osp.join(snapshot_dir, osp.basename(evaluator.prediction_dir)))
    #         shutil.move(snapshot_dir, snapshot_dir + "_{:.3f}".format(accuracy * 100))

if __name__ == "__main__":
    main()

