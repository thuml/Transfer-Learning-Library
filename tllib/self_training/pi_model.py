"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Callable, Optional
import numpy as np
import torch
from torch import nn as nn


def sigmoid_warm_up(current_epoch, warm_up_epochs: int):
    """Exponential warm up function from `Temporal Ensembling for Semi-Supervised Learning
    (ICLR 2017) <https://arxiv.org/abs/1610.02242>`_.
    """
    assert warm_up_epochs >= 0
    if warm_up_epochs == 0:
        return 1.0
    else:
        current_epoch = np.clip(current_epoch, 0.0, warm_up_epochs)
        process = 1.0 - current_epoch / warm_up_epochs
        return float(np.exp(-5.0 * process * process))


class ConsistencyLoss(nn.Module):
    r"""
    Consistency loss between two predictions. Given distance measure :math:`D`, predictions :math:`p_1, p_2`,
    binary mask :math:`mask`, the consistency loss is

    .. math::
        D(p_1, p_2) * mask

    Args:
        distance_measure (callable): Distance measure function.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - p1: the first prediction
        - p2: the second prediction
        - mask: binary mask. Default: 1. (use all samples when calculating loss)

    Shape:
        - p1, p2: :math:`(N, C)` where C means the number of classes.
        - mask: :math:`(N, )` where N means mini-batch size.
    """

    def __init__(self, distance_measure: Callable, reduction: Optional[str] = 'mean'):
        super(ConsistencyLoss, self).__init__()
        self.distance_measure = distance_measure
        self.reduction = reduction

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, mask=1.):
        cons_loss = self.distance_measure(p1, p2)
        cons_loss = cons_loss * mask
        if self.reduction == 'mean':
            return cons_loss.mean()
        elif self.reduction == 'sum':
            return cons_loss.sum()
        else:
            return cons_loss


class L2ConsistencyLoss(ConsistencyLoss):
    r"""
    L2 consistency loss. Given two predictions :math:`p_1, p_2` and binary mask :math:`mask`, the
    L2 consistency loss is

    .. math::
        \text{MSELoss}(p_1, p_2) * mask

    """

    def __init__(self, reduction: Optional[str] = 'mean'):
        def l2_distance(p1: torch.Tensor, p2: torch.Tensor):
            return ((p1 - p2) ** 2).sum(dim=1)

        super(L2ConsistencyLoss, self).__init__(l2_distance, reduction)
