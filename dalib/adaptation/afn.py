from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def l2_norm_loss(x: torch.Tensor, delta: float) -> torch.Tensor:
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + delta
    loss = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return loss


class ImageClassifierHead(nn.Module):
    r"""Classifier Head for AFN.

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1000
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1000,
                 dropout_p: Optional[float] = 0.5):
        super(ImageClassifierHead, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        self.head = nn.Linear(bottleneck_dim, num_classes)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.bottleneck(x)
        if self.training:
            f.mul_(math.sqrt(1 - self.dropout_p))
        predictions = self.head(f)
        return predictions, f
