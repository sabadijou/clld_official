import torch.nn.functional as F
import torch.nn as nn
import torch


class CrossCorrelation(nn.Module):
    '''
    The implementation of cross-correlation which is employed by the CLoS algorithm.
    source: https://github.com/huanglianghua/siamfc-pytorch.git
    Note that the out_scale hyper-parameter is used to scale the logits to around 0~10.
    Please refer to this issue for more details: https://github.com/huanglianghua/siamfc-pytorch/issues/24
    '''
    def __init__(self, out_scale=0.001):
        super(CrossCorrelation, self).__init__ ()
        self.out_scale = out_scale

    def forward(self, x, z):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = torch.reshape(x, shape=(nz * c, h, w)).unsqueeze(0)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out


class SimilarityLoss(nn.Module):
    def __init__(self, alpha=3, out_scale=0.001):
        super(SimilarityLoss, self).__init__()
        self.cross_correlation_opt = CrossCorrelation(out_scale=out_scale)
        self.cos = nn.CosineSimilarity(dim=1)
        self.alpha = alpha

    def forward(self, y, y_prime):
        map_1 = self.apply_cc(y, y_prime)
        map_2 = self.apply_cc(y_prime, y)
        loss = self.cos(map_1, map_2)
        return loss.mean()

    def apply_cc(self, flat, depth):
        tensor_list = []
        for row in range(0, flat.size(2), self.alpha):
            for col in range(0, flat.size(3), self.alpha):
                d = depth[:, :, row:row + self.alpha, col:col + self.alpha]
                if not d.shape[2] % self.alpha == 0:
                    d = F.pad(d, pad=(0, 0, 0, self.alpha - depth.shape[2] % self.alpha), mode='constant', value=0)
                if not d.shape[3] % self.alpha == 0:
                    d = F.pad(d, pad=(0, self.alpha - depth.shape[3] % self.alpha, 0, 0), mode='constant', value=0)
                out = self.cross_correlation_opt(flat, d)
                tensor_list.append(out.squeeze(1))
        return torch.stack(tensors=tensor_list, dim=1)