from utils.losses import grouping, instance, similarity
from datasets.imagenet import CLoSDataSet
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from models.clos import CLoS
from einops import rearrange
from torch.optim import SGD
from torchlars import LARS
import torch.nn as nn
import torch
import math
import os


#To_Do : resume from a checkpoint
#To Do : add to arg


def main_worker(gpu, gpus_per_node, cfg):
    cfg.device = gpu
    if cfg.distributed_training['distributed']:
        if cfg.distributed_training['env_ip'] == 'env://' and \
                cfg.distributed_training['rank'] == -1:
            cfg.distributed_training['rank'] = int(os.environ["RANK"])

        if cfg.distributed_training['multiprocessing_distributed']:
            cfg.distributed_training['rank'] = cfg.distributed_training['rank'] * \
                                               gpus_per_node +  cfg.device

            dist.init_process_group(backend=cfg.distributed_training['backend'],
                                    init_method=cfg.distributed_training['env_ip'],
                                    world_size=cfg.distributed_training['world_size'],
                                    rank=cfg.distributed_training['rank'])

    model = CLoS(cfg)
    if cfg.distributed_training['distributed']:
        if cfg.device is not None:
            torch.cuda.set_device(cfg.device)
            model.cuda(cfg.device)
            cfg.training_parameters['batch_size'] = int(cfg.training_parameters['batch_size'] / gpus_per_node)
            cfg.training_parameters['workers'] = int((cfg.training_parameters['workers'] + gpus_per_node - 1) / gpus_per_node)
            sync_bn_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.device])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.device is not None:
        torch.cuda.set_device(cfg.device)
        model = model.cuda(cfg.device)
    else:
        raise NotImplementedError('only DDP is supported.')

    cfg.lr = cfg.training_parameters['init_lr'] * cfg.training_parameters['batch_size'] / 256
    optimizer = LARS(optimizer=SGD(model.parameters(),
                                    lr=cfg.lr,
                                    weight_decay=cfg.training_parameters['weight_decay']),
                          eps=cfg.training_parameters['optim_eps'])
    cudnn.benchmark = True
    loader, train_sampler = setup_dataset(cfg)
    for epoch in range(cfg.resume['start_epoch'],
                       cfg.training_parameters['num_epochs']):

        train_sampler.set_epoch(epoch)
        lr_scheduler(epoch, optimizer, cfg)
        train_step(model, loader, optimizer, cfg, gpu)
        dist.barrier()

def train_step(model, loader, optimizer, cfg, gpu):
    model.train()
    loss_1 = grouping.GroupingLoss()
    loss_2 = similarity.SimilarityLoss(alpha=cfg.training_parameters['alpha'],
                                       out_scale=cfg.training_parameters['out_scale'])
    loss_3 = instance.InstanceLoss()
    flatten = lambda t: rearrange(t, 'b c h w -> b c (h w)')
    for _iter, (images, targets) in enumerate(loader):
        images[0], images[1] = images[0].cuda(gpu, non_blocking=True), \
                               images[1].cuda(gpu, non_blocking=True)

        yi, xj_moment, base_i = model(images[0], images[1])
        yj, xi_moment, base_j = model(images[1], images[0])

        base_A_matrix, moment_A_matrix = targets[0].cuda(gpu), \
                                         targets[1].cuda(gpu)

        loss1 = cfg.training_parameters['coeff_lamda'] * \
                loss_1(yi, xj_moment, base_A_matrix) + loss_1(yj, xi_moment, moment_A_matrix)

        target_proj_pixel_one, target_proj_pixel_two = list(map(flatten, (xj_moment, xi_moment)))
        proj_pixel_one, proj_pixel_two = list(map(flatten, (base_i, base_j)))
        loss2 = cfg.training_parameters['coeff_beta'] * loss_2(yi, yj)

        loss3 = cfg.training_parameters['coeff_Xi'] * \
                (loss_3(proj_pixel_one, target_proj_pixel_two)
                 + loss_3(proj_pixel_two, target_proj_pixel_one)).mean()

        loss_clos = loss1 + loss2 + loss3
        print(loss_clos.item(), loss1.item(), loss2.item(), loss3.item())
        optimizer.zero_grad()
        loss_clos.backward()
        optimizer.step()


def lr_scheduler(epoch, optimizer, cfg):
    cfg.lr *= 0.5 * (1. + math.cos(math.pi * epoch /
                                   cfg.training_parameters['num_epochs']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.lr


def setup_dataset(cfg):
    dataset = CLoSDataSet(cfg=cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    return DataLoader(dataset,
                      batch_size=cfg.training_parameters['batch_size'],
                      shuffle=(train_sampler is None),
                      num_workers=cfg.training_parameters['workers'],
                      pin_memory=True,
                      sampler=train_sampler,
                      drop_last=True), train_sampler