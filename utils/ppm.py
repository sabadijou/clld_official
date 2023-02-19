from torch import pow, einsum
import torch.nn as nn


class PPM(nn.Module):
    def __init__(self, input_channels=256, out_channels=256):
        super(PPM, self).__init__()
        self.linear = nn.Conv2d(in_channels=input_channels,
                                out_channels=out_channels,
                                kernel_size=(1, 1))

        self.relu = nn.ReLU()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = x.view(b, c, h, w, 1, 1)
        x2 = x.view(b, c, 1, 1, h, w)
        similarity_vector = pow(self.relu(self.cos_sim(x1, x2)), 2)
        transformed_x = self.linear(x)
        return einsum('bijhw, bchw -> bcij', similarity_vector, transformed_x)
