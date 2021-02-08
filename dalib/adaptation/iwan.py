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


class AutomaticUpdateClassWeightModule(object):
    r"""
    Calculating class weight based on the output of domain discriminator and ground truth labels
    """

    def __init__(self, num_classes: int, partial_classes_index: Optional[List[int]], device: torch.device):
        self.device = device
        self.class_weight_sum = torch.zeros(num_classes).to(device)
        self.class_weight_count = torch.zeros(num_classes).to(device)
        self.partial_classes_index = partial_classes_index
        self.non_partial_classes_index = [c for c in range(num_classes) if c not in partial_classes_index]

    def update_weight(self, weight: torch.Tensor, labels: torch.Tensor):
        weight = weight.view_as(labels)
        for idx, c in enumerate(labels):
            self.class_weight_sum[c] += weight[idx]
            self.class_weight_count[c] += 1

    def get_partial_classes_weight(self):
        """
        Get class weight averaged on the partial classes and non-partial classes respectively.

        .. warning::

            This function is just for debugging, since in real-world dataset, we have no access to the index of \
            partial classes and this function will throw an error when `partial_classes_index` is None.
        """
        class_weight = self.class_weight_sum / (
                self.class_weight_count + (self.class_weight_count == 0))  # avoid division by zero
        return torch.mean(class_weight[self.partial_classes_index]), torch.mean(
            class_weight[self.non_partial_classes_index])
