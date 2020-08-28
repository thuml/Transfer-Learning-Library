from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dalib.modules.classifier import Classifier as ClassifierBase
from ._util import entropy


__all__ = ['MinimumClassConfusionLoss', 'ImageClassifier']


class MinimumClassConfusionLoss(nn.Module):
    r"""The `Minimum Class Confusion Loss Loss <https://arxiv.org/abs/1912.03699>`_

        Minimum Class Confusion is a non-adversarial DA method without explicitly deploying domain alignment.
        It can handle four existing scenarios: Closed-Set, Partial-Set, Multi-Source, and Multi-Target DA by minimizing
        class confusion on the target domain. Also, it can be used as a general regularizer that is orthogonal and complementary
        to a variety of existing DA methods.

        Parameters:
            - **temperature** (float): the temperature scaling when calculating predictions.

        Inputs:
            - **logits** (tensor): unnormalized classifier predictions on target domain

        Shape:
            - logits: :math:`(minibatch, C)` where C means the number of classes.
            - Output: scalar by default

        Examples::
            >>> loss = MinimumClassConfusionLoss(temperature=2.)
            >>> # logits output from target domain
            >>> logits = torch.randn(batch_size, num_classes)
            >>> output = loss(logits)
    """
    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
