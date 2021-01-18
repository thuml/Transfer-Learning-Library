from typing import Optional, List, Dict
import torch
import torch.nn as nn
import numpy as np

from dalib.modules.classifier import Classifier as ClassifierBase


class ImageClassifier(ClassifierBase):
    r"""The Image Classifier for `Importance Weighted Adversarial Nets for Partial Domain Adaptation <https://arxiv.org/abs/1803.09210>`_
    """
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.features_only = False
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def set_features_only(self, option: bool):
        """Set self.features_only according to input option
        Inputs:
            **option**(bool): option = 'Ture' means we only optimize feature extractors(backbone, bottleneck)
                otherwise we optimize all parameters(backbone, bottleneck, head)
        """
        self.features_only = option

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        if self.features_only:
            params = [
                {"params": self.backbone.parameters(), "lr": 0.1 if self.finetune else 1.},
                {"params": self.bottleneck.parameters(), "lr": 1.},
            ]
        else:
            params = [
                {"params": self.backbone.parameters(), "lr": 0.1 if self.finetune else 1.},
                {"params": self.bottleneck.parameters(), "lr": 1.},
                {"params": self.head.parameters(), "lr": 1.},
            ]
        return params


class ImageClassifierHead(nn.Module):
    r"""Classifier Head for Importance Weighted Adversarial Nets.
        Parameters:
            - **in_features** (int): Dimension of input features
            - **num_classes** (int): Number of classes
            - **bottleneck_dim** (int, optional): Feature dimension of the bottleneck layer. Default: 1024

        Shape:
            - Inputs: :math:`(minibatch, F)` where F = `in_features`.
            - Output: :math:`(minibatch, C)` where C = `num_classes`.
        """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1024):
        super(ImageClassifierHead, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)
