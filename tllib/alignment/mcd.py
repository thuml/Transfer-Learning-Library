"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import torch.nn as nn
import torch


def classifier_discrepancy(predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:
    r"""The `Classifier Discrepancy` in
    `Maximum Classiﬁer Discrepancy for Unsupervised Domain Adaptation (CVPR 2018) <https://arxiv.org/abs/1712.02560>`_.

    The classfier discrepancy between predictions :math:`p_1` and :math:`p_2` can be described as:

    .. math::
        d(p_1, p_2) = \dfrac{1}{K} \sum_{k=1}^K | p_{1k} - p_{2k} |,

    where K is number of classes.

    Args:
        predictions1 (torch.Tensor): Classifier predictions :math:`p_1`. Expected to contain raw, normalized scores for each class
        predictions2 (torch.Tensor): Classifier predictions :math:`p_2`
    """
    return torch.mean(torch.abs(predictions1 - predictions2))


def entropy(predictions: torch.Tensor) -> torch.Tensor:
    r"""Entropy of N predictions :math:`(p_1, p_2, ..., p_N)`.
    The definition is:

    .. math::
        d(p_1, p_2, ..., p_N) = -\dfrac{1}{K} \sum_{k=1}^K \log \left( \dfrac{1}{N} \sum_{i=1}^N p_{ik} \right)

    where K is number of classes.

    .. note::
        This entropy function is specifically used in MCD and different from the usual :meth:`~tllib.modules.entropy.entropy` function.

    Args:
        predictions (torch.Tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
    """
    return -torch.mean(torch.log(torch.mean(predictions, 0) + 1e-6))


class ImageClassifierHead(nn.Module):
    r"""Classifier Head for MCD.

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024

    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1024, pool_layer=None):
        super(ImageClassifierHead, self).__init__()
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool_layer(inputs))