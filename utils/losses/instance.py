import torch.nn.functional as F
import torch.nn as nn


class InstanceLoss(nn.Module):
    def __init__(self):
        super(InstanceLoss, self).__init__()

    def forward(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        instance_loss = 2 - 2 * (x * y).sum(dim=-1)
        return instance_loss