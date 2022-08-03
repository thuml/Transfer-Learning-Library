"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from tllib.modules.classifier import Classifier as ClassifierBase


class ClassBalanceLoss(nn.Module):
    r"""
    Class balance loss that penalises the network for making predictions that exhibit large class imbalance.
    Given predictions :math:`p` with dimension :math:`(N, C)`, we first calculate
    the mini-batch mean per-class probability :math:`p_{mean}` with dimension :math:`(C, )`, where

    .. math::
        p_{mean}^j = \frac{1}{N} \sum_{i=1}^N p_i^j

    Then we calculate binary cross entropy loss between :math:`p_{mean}` and uniform probability vector :math:`u` with
    the same dimension where :math:`u^j` = :math:`\frac{1}{C}`

    .. math::
        loss = \text{BCELoss}(p_{mean}, u)

    Args:
        num_classes (int): Number of classes

    Inputs:
        - p (tensor): predictions from classifier

    Shape:
        - p: :math:`(N, C)` where C means the number of classes.
    """

    def __init__(self, num_classes):
        super(ClassBalanceLoss, self).__init__()
        self.uniform_distribution = torch.ones(num_classes) / num_classes

    def forward(self, p: torch.Tensor):
        return F.binary_cross_entropy(p.mean(dim=0), self.uniform_distribution.to(p.device))


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
