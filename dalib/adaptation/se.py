from typing import Optional
import torch.nn as nn

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
