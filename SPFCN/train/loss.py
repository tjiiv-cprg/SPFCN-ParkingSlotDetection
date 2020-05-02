import torch
from torch import nn


class FocalLossWithTrigonometric(nn.Module):
    def __init__(self, threshold=0.1, pos=2, neg=0.1, cos=1, sin=1, const=4):
        nn.Module.__init__(self)
        self.threshold = threshold
        self._pos = -pos
        self._neg = -neg
        self._cos = cos
        self._sin = sin
        self._const = const

    def forward(self, mark, direction, gt):
        mark_gt = gt[:, 0:1]
        num_pos = mark_gt.eq(1).float().sum()
        pos_gt = mark_gt.gt(self.threshold)
        mark_pos = mark[pos_gt]
        mark_pos_loss = torch.pow(mark_gt[pos_gt] - mark_pos, 2) * torch.log(mark_pos)
        mark_pos_loss_value = self._pos * mark_pos_loss.float().sum() / num_pos
        # print(mark_pos_loss_value)

        neg_gt = mark_gt.le(self.threshold)
        mark_neg = mark[neg_gt]
        mark_neg_loss = torch.pow(mark_neg, 2) * torch.log(1 - mark_neg) * torch.pow(1 - mark_gt[neg_gt], 4)
        mark_neg_loss_value = self._neg * mark_neg_loss.float().sum()
        # print(mark_neg_loss_value)

        cos_prediction = direction[:, 0:1][pos_gt]
        cos_gt = gt[:, 1:2][pos_gt]
        cos_loss = torch.pow(cos_gt - cos_prediction, 2)
        cos_loss_value = self._cos * cos_loss.float().sum() / num_pos
        # print(cos_loss_value)

        sin_prediction = direction[:, 1:2][pos_gt]
        sin_gt = gt[:, 2:3][pos_gt]
        sin_loss = torch.pow(sin_gt - sin_prediction, 2)
        sin_loss_value = self._sin * sin_loss.float().sum() / num_pos
        # print(sin_loss_value)

        const_loss = torch.pow((1 - torch.pow(cos_prediction, 2) - torch.pow(sin_prediction, 2)), 2)
        const_loss_value = self._const * const_loss.float().sum() / num_pos
        # print(const_loss_value)

        return mark_pos_loss_value + mark_neg_loss_value + cos_loss_value + sin_loss_value + const_loss_value
