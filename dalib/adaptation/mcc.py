from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dalib.modules.grl import WarmStartGradientReverseLayer
from dalib.modules.classifier import Classifier as ClassifierBase
from ._util import binary_accuracy

__all__ = ['ConditionalDomainAdversarialLoss', 'ImageClassifier']


class MinimumClassConfusionLoss(nn.Module):

    def __init__(self, temperature):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, g_t: torch.Tensor) -> torch.Tensor:
        train_bs, class_num = g_t.size(0), g_t.size(1)
        g_t_temp = g_t / self.temperature
        g_t_temp_softmax = nn.Softmax(dim=1)(g_t_temp)
        target_entropy_weight = entropy(g_t_temp_softmax).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
        c_matrix = g_t_temp_softmax.mul(target_entropy_weight.view(-1,1)).transpose(1,0).mm(g_t_temp_softmax)
        c_matrix = c_matrix / torch.sum(c_matrix, dim=1)
        mcc_loss = (torch.sum(c_matrix) - torch.trace(c_matrix)) / class_num
        return mcc_loss


def entropy(predictions: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    return H.sum(dim=1)


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
