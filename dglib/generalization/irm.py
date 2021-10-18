"""
Modified from https://github.com/facebookresearch/DomainBed
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class InvariancePenaltyLoss(nn.Module):
    r"""Invariance Penalty Loss from `Invariant Risk Minimization <https://arxiv.org/pdf/1907.02893.pdf>`_.
    We adopt implementation from `DomainBed <https://github.com/facebookresearch/DomainBed>`_. Given classifier
    output :math:`y` and ground truth :math:`labels`, we split :math:`y` into two parts :math:`y_1, y_2`, corresponding
    labels are :math:`labels_1, labels_2`. Next we calculate cross entropy loss with respect to a dummy classifier
    :math:`w`, resulting in :math:`grad_1, grad_2` . Invariance penalty is then :math:`grad_1*grad_2`.

    Inputs:
        - y: predictions from model
        - labels: ground truth

    Shape:
        - y: :math:`(N, C)` where C means the number of classes.
        - labels: :math:`(N, )` where N mean mini-batch size
    """

    def __init__(self):
        super(InvariancePenaltyLoss, self).__init__()
        self.scale = torch.tensor(1.).requires_grad_()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_1 = F.cross_entropy(y[::2] * self.scale, labels[::2])
        loss_2 = F.cross_entropy(y[1::2] * self.scale, labels[1::2])
        grad_1 = autograd.grad(loss_1, [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [self.scale], create_graph=True)[0]
        penalty = torch.sum(grad_1 * grad_2)
        return penalty
