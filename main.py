from utils.config import convert_config
from utils.recorder import Recorder
from runner.trainer import *
import configs.clos as cfg
import argparse
import random
import torch

# To do : Add recorder


def main():
    args = parse_args()
    assert len(args.gpus_id) > 1, 'This repository does not support single GPU training'
    cfg.dataset['path'] = args.dataset_path
    cfg.encoder['backbone'] = args.encoder
    cfg.work_dirs['experiments'] = args.work_dir
    cfg.training_parameters['alpha'] = args.alpha
    cfg.distributed_training['gpus_idx'] = args.gpus_id

    random.seed(cfg.training_parameters['seed'])
    torch.manual_seed(cfg.training_parameters['seed'])
    cudnn.deterministic = True
    gpus_per_node = len(cfg.distributed_training['gpus_idx'])
    cfg.distributed_training['world_size'] = gpus_per_node * args.world_size
    args = convert_config(args, cfg)
    log_recorder = Recorder(args)
    torch.multiprocessing.spawn(main_worker, nprocs=gpus_per_node, args=(gpus_per_node, log_recorder, args))

def parse_args():
    parser = argparse.ArgumentParser(description='Train CLoS')
    parser.add_argument(
        '--dataset_path', type=str, default=r'/media/ali/ssd/Imagenet/train',
        help='path to image folder')

    parser.add_argument(
        '--work_dir', type=str, default=r'work_dir',
        help='path to work dir')

    parser.add_argument(
        '--encoder', default='resnet50',
        help='choose an encoder for training. ["resnet18", "resnet34", '
             '"resnet50", "resnet101", "resnet152", "resnext50_32x4d", '
             '"resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2"]')

    parser.add_argument(
        '--alpha', type=int, default=1,
        help='initialize similarity loss patch size. 1, 2, and, 3 are recommended to train resnet50 and resnet34.')

    parser.add_argument('--world_size', nargs='+', type=int, default=1)
    parser.add_argument('--gpus_id', nargs='+', type=int, default=[0, 1])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()