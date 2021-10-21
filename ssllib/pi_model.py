"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.mse_loss(input_softmax, target_softmax, reduction='sum')


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2) ** 2) / num_classes


class SoftmaxMSELoss(nn.Module):
    r"""
    The Softmax MSE Loss of two predictions :math:`z, z'` can be described as:
    .. math::
        p_i(c) = \dfrac{\exp z(c)}{\sum_{k=1}^{C}\exp(z(k))},\ c=1,..,C\\
        p'_i(c) = \dfrac{\exp z'(c)}{\sum_{k=1}^{C}\exp(z'_i(k))},\ c=1,..,C\\
        L = \sum_{i=1}^{b}\sum_{c=1}^{C}[p_i(c) - p'_i(c)]^2
    where :math:`C` is the number of classes, :math:`p, p'` are probability distribution calculated from the predictions :math:`z, z'`.
    Inputs:
        - input1 (tensor): The prediction of the input with a random data augmentation.
        - input2 (tensor): Another prediction of the input with a random data augmentation.
        - reduction: string. Default: 'sum'.
    Shape:
        - input1: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - input2: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - Output: scalar.
    """

    def __init__(self):
        super(SoftmaxMSELoss, self).__init__()

    def forward(self, input1, input2, reduction='sum'):
        assert input1.size() == input2.size()
        input1_softmax = F.softmax(input1, dim=1)
        input2_softmax = F.softmax(input2, dim=1)
        return F.mse_loss(input1_softmax, input2_softmax, reduction=reduction)


class SoftmaxKLLoss(nn.Module):
    r"""
    The Softmax KL Loss of two predictions :math:`z, z'` can be described as:
    .. math::
        p_i(c) = \dfrac{\exp z(c)}{\sum_{k=1}^{C}\exp(z(k))},\ c=1,..,C \\
        p'_i(c) = \dfrac{\exp z'(c)}{\sum_{k=1}^{C}\exp(z'_i(k))},\ c=1,..,C\\
        L = \sum_{i=1}^{b}\sum_{c=1}^{C}[p_i(c)\log\frac{p_i(c)}{p'_i(c)}]^2
    where :math:`C` is the number of classes, :math:`p, p'` are probability distribution calculated from the predictions :math:`z, z'`.
    Inputs:
        - input1 (tensor): The prediction of the input with a random data augmentation.
        - input2 (tensor): Another prediction of the input with a random data augmentation.
    Shape:
        - input1: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - input2: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - Output: scalar.
    """

    def __init__(self):
        super(SoftmaxKLLoss, self).__init__()

    def forward(self, input1, input2):
        assert input1.size() == input2.size()
        input1_log_softmax = F.log_softmax(input1, dim=1)
        input2_softmax = F.softmax(input2, dim=1)
        return F.kl_div(input1_log_softmax, input2_softmax, size_average=False)
