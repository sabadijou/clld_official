from collections import deque, defaultdict
from .logger import get_logger
import datetime
import torch
import os


class SmoothedValue(object):
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = self.get_work_dir()
        self.log_path = os.path.join(self.work_dir, 'training_log.txt')

        self.logger = get_logger('CLoS', self.log_path)
        self.logger.info('Training {backbone} on {dataset} dataset using CLoS based on the following hyperparameters :'
                         .format(backbone=self.cfg.encoder['backbone'], dataset=self.cfg.dataset['name']))
        for key in self.cfg.training_parameters:
            self.logger.info(key + ' : ' + str(self.cfg.training_parameters[key]))

        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()
        self.max_epoch = self.cfg.training_parameters['num_epochs']
        self.lr = 0.

    def get_work_dir(self):
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        hyper_param_str = '_alpha_%1.0e_b_%d' % (self.cfg.training_parameters['alpha'],
                                                 self.cfg.training_parameters['batch_size'])
        work_dir = os.path.join(self.cfg.work_dirs['path'], now + hyper_param_str)

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        return work_dir

    def update_loss_stats(self, loss_dict):
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        self.logger.info(self)

    def write(self, content):
        with open(self.log_path, 'a+') as f:
            f.write(content)
            f.write('\n')

    def state_dict(self):
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict['step']

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.6f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'lr: {:.6f}', '{}'])
        return recording_state.format(self.epoch, self.lr, loss_state)
