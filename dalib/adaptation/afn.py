from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import math

from common.modules.classifier import Classifier as ClassfierBase


class AdaptiveFeatureNorm(nn.Module):
    r"""
    Stepwise feature norm loss

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
    r"""Basic building block for Image Classifier with structure: FC-BN-ReLU-Dropout order
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
    r"""The Image Classifier for 'Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised
        Domain Adaptation'

    """

    def __init__(self, backbone: nn.Module, num_classes: int, num_blocks: Optional[int] = 1,
                 bottleneck_dim: Optional[int] = 1000, dropout_p: Optional[float] = 0.5, **kwargs):
        assert num_blocks >= 1
        layers = [nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
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

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters()},
            {"params": self.bottleneck.parameters(), "momentum": 0.9},
            {"params": self.head.parameters(), "momentum": 0.9},
        ]
        return params
