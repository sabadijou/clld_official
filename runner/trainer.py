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


#To_Do : resume from a checkpoint
#To Do : add to arg


class Trainer:
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.model = CLoS(self.cfg)
        self.lr = self.cfg.training_parameters['init_lr'] * self.cfg.training_parameters['batch_size'] / 256
        self.loader = self.setup_dataset()
        self.loss_1 = grouping.GroupingLoss()
        self.loss_2 = similarity.SimilarityLoss(alpha=self.cfg.training_parameters['alpha'],
                                                out_scale=self.cfg.training_parameters['out_scale'])
        self.loss_3 = instance.InstanceLoss()

    def main_worker(self, gpus_per_node):
        self.cfg.distributed_training['rank'] = self.cfg.distributed_training['rank'] * \
                                        gpus_per_node + \
                                        self.cfg.distributed_training['gpus_idx']

        dist.init_process_group(backend=self.cfg.distributed_training['backend'],
                                init_method=self.cfg.distributed_training['env_ip'],
                                world_size=self.cfg.distributed_training['world_size'],
                                rank=self.cfg.distributed_training['rank'])


        torch.cuda.set_device(self.cfg.distributed_training['gpus_idx'])
        self.model.cuda(self.cfg.distributed_training['gpus_idx'])

        self.cfg.training_parameters['batch_size'] = int(self.cfg.training_parameters['batch_size'] / gpus_per_node)

        self.cfg.training_parameters['workers'] = int((self.cfg.training_parameters['workers'] + gpus_per_node - 1)
                                                   / gpus_per_node)

        self.sync_bn_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                               device_ids=[self.cfg.distributed_training['gpus_idx']])

        self.optimizer = LARS(optimizer=SGD(self.model.parameters(),
                                            lr=self.lr,
                                            weight_decay=self.cfg.training_parameters['weight_decay']),
                              eps=self.cfg.training_parameters['optim_eps'])
        cudnn.benchmark = True

        for epoch in range(self.cfg.resume['start_epoch'],
                           self.cfg.training_parameters['num_epochs']):

            self.train_sampler.set_epoch(epoch)
            self.lr_scheduler(epoch)

    def train_step(self):
        self.model.train()
        flatten = lambda t: rearrange(t, 'b c h w -> b c (h w)')
        for _iter, (images, targets) in enumerate(self.loader):
            images[0], images[1] = images[0].cuda(self.cfg.distributed_training['gpus_idx'], non_blocking=True), \
                                   images[1].cuda(self.cfg.distributed_training['gpus_idx'], non_blocking=True)

            yi, xj_moment, base_i = self.model(images[0], images[1])
            yj, xi_moment, base_j = self.model(images[1], images[0])

            base_A_matrix, moment_A_matrix = targets[0].cuda(self.cfg.distributed_training['gpus_idx']), \
                                             targets[1].cuda(self.cfg.distributed_training['gpus_idx'])

            loss1 = self.cfg.training_parameters['coeff_lamda'] * \
                    self.loss_1(yi, xj_moment, base_A_matrix) + self.loss_1(yj, xi_moment, moment_A_matrix)

            target_proj_pixel_one, target_proj_pixel_two = list(map(flatten, (xj_moment, xi_moment)))
            proj_pixel_one, proj_pixel_two = list(map(flatten, (base_i, base_j)))

            loss2 = self.cfg.training_parameters['coeff_beta'] * self.loss_2(yi, yj)

            loss3 = self.cfg.training_parameters['coeff_Xi'] * \
                    (self.loss_3(proj_pixel_one, target_proj_pixel_two)
                     + self.loss_3(proj_pixel_two, target_proj_pixel_one)).mean()

            loss_clos = loss1 + loss2 + loss3
            print(loss_clos, loss1, loss2, loss3)
            self.optimizer.zero_grad()
            loss_clos.backward()
            self.optimizer.step()

    def lr_scheduler(self, epoch):
        self.lr *= 0.5 * (1. + math.cos(math.pi * epoch /
                                        self.cfg.training_parameters['num_epochs']))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def setup_dataset(self):
        dataset = CLoSDataSet(cfg=self.cfg)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(dataset,
                          batch_size=self.cfg.training_parameters['batch_size'],
                          shuffle=(self.train_sampler is None),
                          num_workers=self.cfg.training_parameters['workers'],
                          pin_memory=True,
                          sampler=self.train_sampler,
                          drop_last=True)
