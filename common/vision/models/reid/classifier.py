import torch.nn as nn
from torch.nn import init
from common.modules.classifier import Classifier as ClassifierBase


class ReidClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, **kwargs):
        super(ReidClassifier, self).__init__(backbone, num_classes, **kwargs)
        self.feature_bn = nn.BatchNorm1d(self.features_dim)
        self.feature_bn.bias.requires_grad_(False)
        init.constant_(self.feature_bn.weight, 1)
        init.constant_(self.feature_bn.bias, 0)
        self.head = nn.Linear(self.features_dim, num_classes, bias=False)
        init.normal_(self.head.weight, std=0.001)

    def forward(self, x):
        f = self.backbone(x)
        f = self.bottleneck(f)
        bn_f = self.feature_bn(f)
        if not self.training:
            return bn_f
        predictions = self.head(bn_f)
        return predictions, f
