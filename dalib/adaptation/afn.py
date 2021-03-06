from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import math

from dalib.modules.classifier import Classifier as ClassfierBase


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


class ImageClassifier(ClassfierBase):
    r"""The Image Classifier for 'Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised
        Domain Adaptation'

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 1000,
                 dropout_p: Optional[float] = 0.5, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        self.dropout_p = dropout_p
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        f = self.bottleneck(f)
        if self.training:
            f.mul_(math.sqrt(1 - self.dropout_p))
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters()},
            {"params": self.bottleneck.parameters(), "momentum": 0.9},
            {"params": self.head.parameters(), "momentum": 0.9},
        ]
        return params
