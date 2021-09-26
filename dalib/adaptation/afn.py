"""
Modified from https://github.com/jihanyang/AFN
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import math

from common.modules.classifier import Classifier as ClassfierBase


class AdaptiveFeatureNorm(nn.Module):
    r"""
    The `Stepwise Adaptive Feature Norm loss (ICCV 2019) <https://arxiv.org/pdf/1811.07456v2.pdf>`_

    Instead of using restrictive scalar R to match the corresponding feature norm, Stepwise Adaptive Feature Norm
    is used in order to learn task-specific features with large norms in a progressive manner.
    We denote parameters of backbone :math:`G` as :math:`\theta_g`, parameters of bottleneck :math:`F_f` as :math:`\theta_f`
    , parameters of classifier head :math:`F_y` as :math:`\theta_y`, and features extracted from sample :math:`x_i` as
    :math:`h(x_i;\theta)`. Full loss is calculated as follows

    .. math::
        L(\theta_g,\theta_f,\theta_y)=\frac{1}{n_s}\sum_{(x_i,y_i)\in D_s}L_y(x_i,y_i)+\frac{\lambda}{n_s+n_t}
        \sum_{x_i\in D_s\cup D_t}L_d(h(x_i;\theta_0)+\Delta_r,h(x_i;\theta))\\

    where :math:`L_y` denotes classification loss, :math:`L_d` denotes norm loss, :math:`\theta_0` and :math:`\theta`
    represent the updated and updating model parameters in the last and current iterations respectively.

    Args:
        delta (float): positive residual scalar to control the feature norm enlargement.

    Inputs:
        - f (tensor): feature representations on source or target domain.

    Shape:
        - f: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    Examples::

        >>> adaptive_feature_norm = AdaptiveFeatureNorm(delta=1)
        >>> f_s = torch.randn(32, 1000)
        >>> f_t = torch.randn(32, 1000)
        >>> norm_loss = adaptive_feature_norm(f_s) + adaptive_feature_norm(f_t)
    """

    def __init__(self, delta):
        super(AdaptiveFeatureNorm, self).__init__()
        self.delta = delta

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        radius = f.norm(p=2, dim=1).detach()
        assert radius.requires_grad == False
        radius = radius + self.delta
        loss = ((f.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss


class Block(nn.Module):
    r"""
    Basic building block for Image Classifier with structure: FC-BN-ReLU-Dropout.
    We use :math:`L_2` preserved dropout layers.
    Given mask probability :math:`p`, input :math:`x_k`, generated mask :math:`a_k`,
    vanilla dropout layers calculate

    .. math::
        \hat{x}_k = a_k\frac{1}{1-p}x_k\\

    While in :math:`L_2` preserved dropout layers

    .. math::
        \hat{x}_k = a_k\frac{1}{\sqrt{1-p}}x_k\\

    Args:
        in_features (int): Dimension of input features
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1000
        dropout_p (float, optional): dropout probability. Default: 0.5
    """

    def __init__(self, in_features: int, bottleneck_dim: Optional[int] = 1000, dropout_p: Optional[float] = 0.5):
        super(Block, self).__init__()
        self.fc = nn.Linear(in_features, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.fc(x)
        f = self.bn(f)
        f = self.relu(f)
        f = self.dropout(f)
        if self.training:
            f.mul_(math.sqrt(1 - self.dropout_p))
        return f


class ImageClassifier(ClassfierBase):
    r"""
    ImageClassifier for AFN.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        num_blocks (int, optional): Number of basic blocks. Default: 1
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1000
        dropout_p (float, optional): dropout probability. Default: 0.5
    """

    def __init__(self, backbone: nn.Module, num_classes: int, num_blocks: Optional[int] = 1,
                 bottleneck_dim: Optional[int] = 1000, dropout_p: Optional[float] = 0.5, **kwargs):
        assert num_blocks >= 1
        layers = [nn.Sequential(
            Block(backbone.out_features, bottleneck_dim, dropout_p)
        )]
        for _ in range(num_blocks - 1):
            layers.append(Block(bottleneck_dim, bottleneck_dim, dropout_p))
        bottleneck = nn.Sequential(*layers)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        # init parameters for bottleneck and head
        for m in self.bottleneck.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.01)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.normal_(0.0, 0.01)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.normal_(0.0, 0.01)

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters()},
            {"params": self.bottleneck.parameters(), "momentum": 0.9},
            {"params": self.head.parameters(), "momentum": 0.9},
        ]
        return params
