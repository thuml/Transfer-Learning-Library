from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from dalib.modules.classifier import Classifier as ClassifierBase
from dalib.modules.grl import GradientReverseLayer


class UnknownClassBinaryCrossEntropy(nn.Module):
    r"""
    Binary cross entropy loss to make a boundary for unknown samples, proposed by
    `Open Set Domain Adaptation by Backpropagation <https://arxiv.org/abs/1804.10427>`_.

    Given a sample on target domain :math:`x_t` and its classifcation outputs :math:`y`, the binary cross entropy
    loss is defined as

    .. math::
        L_{adv}(x_t) = -t log(p(y=C+1|x_t)) - (1-t)log(1-p(y=C+1|x_t))

    where t is a hyper-parameter and C is the number of known classes.

    Parameters:
        - **t** (float): Predefined hyper-parameter. Default: 0.5

    Inputs: y
        - **y** (tensor): classification outputs (before softmax).

    Shape:
        - y: :math:`(minibatch, C+1)`  where C is the number of known classes.
        - Outputs: scalar

    """
    def __init__(self, t=0.5):
        super(UnknownClassBinaryCrossEntropy, self).__init__()
        self.t = t

    def forward(self, y):
        # y : N x (C+1)
        softmax_output = F.softmax(y, dim=1)
        unknown_class_prob = softmax_output[:, -1].contiguous().view(-1, 1)
        known_class_prob = 1. - unknown_class_prob

        unknown_target = torch.ones((y.size(0), 1)).to(y.device) * self.t
        known_target = 1. - unknown_target
        return - torch.mean(unknown_target * torch.log(unknown_class_prob + 1e-6)) \
               - torch.mean(known_target * torch.log(known_class_prob + 1e-6))


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)
        self.grl = GradientReverseLayer()

    def forward(self, x: torch.Tensor, grad_reverse: Optional[bool] = False):
        features = self.backbone(x)
        features = self.bottleneck(features)
        if grad_reverse:
            features = self.grl(features)
        outputs = self.head(features)
        return outputs, features



