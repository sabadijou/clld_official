from .encoder_wrapper import EncoderWrapper
from utils.ppm import PPM
import torch.nn as nn
import numpy as np
import torch


class CLLD(nn.Module):
    def __init__(self, cfg):
        super(CLLD, self).__init__()
        self.cfg = cfg
        self.moment_iter = 0

        self.moment = self.cfg.training_parameters['encoder_momentum']

        self.base_encoder = EncoderWrapper(cfg.encoder)()

        self.moment_encoder = EncoderWrapper(cfg.encoder)()

        self.ppm_module = PPM(input_channels=self.cfg.ppm_module['input_channels'],
                              out_channels=self.cfg.ppm_module['output_channels'])

        for param_base, param_moment in zip(self.base_encoder.parameters(),
                                            self.moment_encoder.parameters()):
            param_moment.data.copy_(param_base.data)
            param_moment.requires_grad = False


    def forward(self, view_1, view_2):
        base = self.base_encoder(view_1)
        y = self.ppm_module(base)

        with torch.no_grad():
            self._momentum_update()
            y_prime = self.moment_encoder(view_2)
            self._momentum_scaling()

        return y, y_prime, base

    @torch.no_grad()
    def _momentum_update(self):
        for param_base, param_moment in zip(self.base_encoder.parameters(), self.moment_encoder.parameters()):
            param_moment.data = param_moment.data * self.moment + param_base.data * (1. - self.moment)

    def _momentum_scaling(self):
        self.moment += (np.sin(np.pi / 2 *
                               self.moment_iter / self.cfg.training_parameters['num_epochs']))\
                       * (1 - self.moment)
        self.moment_iter += 1