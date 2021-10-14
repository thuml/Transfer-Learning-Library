"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from common.modules.classifier import Classifier as ClassifierBase


class ImageClassifier(ClassifierBase):
    """ImageClassifier specific for reproducing results of `DomainBed <https://github.com/facebookresearch/DomainBed>`_.
    You are free to freeze all `BatchNorm2d` layers and insert one additional `Dropout` layer, this can achieve better
    results for some datasets like PACS but may be worse for others.

    Args:
        backbone (torch.nn.Module): Any backbone to extract features from data
        num_classes (int): Number of classes
        freeze_bn (bool, optional): whether to freeze all `BatchNorm2d` layers. Default: False
        dropout_p (float, optional): dropout ratio for additional `Dropout` layer, this layer is only used when `freeze_bn` is True. Default: 0.1
    """

    def __init__(self, backbone: nn.Module, num_classes: int, freeze_bn: Optional[bool] = False,
                 dropout_p: Optional[float] = 0.1, **kwargs):
        super(ImageClassifier, self).__init__(backbone, num_classes, **kwargs)
        self.freeze_bn = freeze_bn
        if freeze_bn:
            self.feature_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        if self.freeze_bn:
            f = self.feature_dropout(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def train(self, mode=True):
        super(ImageClassifier, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
