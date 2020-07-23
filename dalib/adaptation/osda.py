from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from dalib.modules.grl import WarmStartGradientReverseLayer
from dalib.modules.classifier import Classifier as ClassifierBase
from dalib.modules.grl import GradientReverseLayer
from ._util import binary_accuracy


class UnknownClassBinaryCrossEntropy(nn.Module):
    def __init__(self, t=0.5):
        super(UnknownClassBinaryCrossEntropy, self).__init__()
        self.t = t

    def forward(self, output):
        # x : N x (C+1)
        # softmax_output = F.softmax(output, dim=1)
        # unknown_class_prob = softmax_output[:, -1].contiguous().view(-1, 1) + 1e-6
        # known_class_prob = 1. - unknown_class_prob
        #
        # unknown_target = torch.ones((output.size(0), 1)).to(output.device) * self.t
        # known_target = 1. - unknown_target
        # return F.binary_cross_entropy(torch.cat([known_class_prob, unknown_class_prob]),
        #                               torch.cat([known_target, unknown_target]))
        softmax_output = F.softmax(output, dim=1)
        unknown_class_prob = softmax_output[:, -1].contiguous().view(-1, 1)
        known_class_prob = 1. - unknown_class_prob

        unknown_target = torch.ones((output.size(0), 1)).to(output.device) * self.t
        known_target = 1. - unknown_target
        return - torch.mean(unknown_target * torch.log(unknown_class_prob + 1e-6)) \
               - torch.mean(known_target * torch.log(known_class_prob + 1e-6))


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
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



