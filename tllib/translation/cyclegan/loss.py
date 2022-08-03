"""
Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn
import torch


class LeastSquaresGenerativeAdversarialLoss(nn.Module):
    """
    Loss for `Least Squares Generative Adversarial Network (LSGAN) <https://arxiv.org/abs/1611.04076>`_

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - prediction (tensor): unnormalized discriminator predictions
        - real (bool): if the ground truth label is for real images or fake images. Default: true

    .. warning::
        Do not use sigmoid as the last layer of Discriminator.

    """
    def __init__(self, reduction='mean'):
        super(LeastSquaresGenerativeAdversarialLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, prediction, real=True):
        if real:
            label = torch.ones_like(prediction)
        else:
            label = torch.zeros_like(prediction)
        return self.mse_loss(prediction, label)


class VanillaGenerativeAdversarialLoss(nn.Module):
    """
    Loss for `Vanilla Generative Adversarial Network <https://arxiv.org/abs/1406.2661>`_

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - prediction (tensor): unnormalized discriminator predictions
        - real (bool): if the ground truth label is for real images or fake images. Default: true

    .. warning::
        Do not use sigmoid as the last layer of Discriminator.

    """
    def __init__(self, reduction='mean'):
        super(VanillaGenerativeAdversarialLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, prediction, real=True):
        if real:
            label = torch.ones_like(prediction)
        else:
            label = torch.zeros_like(prediction)
        return self.bce_loss(prediction, label)


class WassersteinGenerativeAdversarialLoss(nn.Module):
    """
    Loss for `Wasserstein Generative Adversarial Network <https://arxiv.org/abs/1701.07875>`_

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - prediction (tensor): unnormalized discriminator predictions
        - real (bool): if the ground truth label is for real images or fake images. Default: true

    .. warning::
        Do not use sigmoid as the last layer of Discriminator.

    """
    def __init__(self, reduction='mean'):
        super(WassersteinGenerativeAdversarialLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, prediction, real=True):
        if real:
            return -prediction.mean()
        else:
            return prediction.mean()