from typing import Optional, Tuple
import torch
import torch.nn as nn
from common.modules.classifier import Classifier as ClassifierBase


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, freeze_bn: Optional[bool] = False,
                 dropout_p: Optional[float] = 0.1, **kwargs):
        super(ImageClassifier, self).__init__(backbone, num_classes, **kwargs)
        self.freeze_bn = freeze_bn
        if freeze_bn:
            self.feature_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        f = self.bottleneck(f)
        if self.freeze_bn:
            f = self.feature_dropout(f)
        predictions = self.head(f)
        return predictions, f

    def train(self, mode=True):
        super(ImageClassifier, self).train(mode)
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
