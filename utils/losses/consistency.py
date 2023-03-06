import torch.nn as nn
import torch


def get_cosine_similarity(base, moment, dim=1):
    base = base.view(base.shape[0], base.shape[1], 1, -1)
    moment = moment.view(moment.shape[0], moment.shape[1], -1, 1)
    cos = nn.CosineSimilarity(dim=dim, eps=1e-6)
    cos_sim = cos(base, moment)
    return cos_sim


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, en_base, en_momentum, matrix):
        distance = get_cosine_similarity(en_base, en_momentum)
        matrix = matrix.type(torch.BoolTensor).cuda()
        loss = - distance.masked_select(matrix).mean()
        return loss