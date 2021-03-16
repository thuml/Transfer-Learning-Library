from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.modules.classifier import Classifier as ClassifierBase


def ema_model_update(model, ema_model, alpha):
    """Exponential moving average of model parameters.
    Args:
        model (nn.Module): model being trained.
        ema_model (nn.Module): ema of the model.
        alpha (float): ema decay rate.
    """
    one_minus_alpha = 1 - alpha
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data * one_minus_alpha)


def consistent_loss(y: torch.Tensor, y_teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Consistent loss between model output y and teacher output y_teacher
    """
    cons_loss = ((y - y_teacher) ** 2).sum(dim=1)
    cons_loss = (cons_loss * mask).mean()
    return cons_loss


def class_balance_loss(y: torch.Tensor, mask) -> torch.Tensor:
    """Class balance loss
    """
    class_distribution = y.mean(dim=0)
    num_classes = y.shape[1]
    uniform_distribution = torch.ones(num_classes) / num_classes
    uniform_distribution = uniform_distribution.to(class_distribution.device)
    balance_loss = F.binary_cross_entropy(class_distribution, uniform_distribution)
    balance_loss = mask.mean() * balance_loss
    return balance_loss


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
