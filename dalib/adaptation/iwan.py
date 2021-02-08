from typing import Optional, List, Dict
import torch
import torch.nn as nn

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
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)


class DomainDiscriminator(nn.Module):
    r"""Domain discriminator model from
        `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_

        Distinguish whether the input features come from the source domain or the target domain.
        The source domain label is 1 and the target domain label is 0.

        .. note::
            **Batch Normalization** is **not** used here

        Args:
            in_feature (int): dimension of the input feature
            hidden_size (int): dimension of the hidden features

        Shape:
            - Inputs: (minibatch, `in_feature`)
            - Outputs: :math:`(minibatch, 1)`
        """

    def __init__(self, in_feature: int, hidden_size: int):
        super(DomainDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]
