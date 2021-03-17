from typing import Optional, ClassVar, Sequence
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.modules.classifier import Classifier as ClassifierBase
from dalib.translation.cyclegan.util import set_requires_grad


class ConsistentLoss(nn.Module):
    """Consistent loss between model output y and teacher output y_teacher
    """
    def __init__(self, reduction='mean'):
        super(ConsistentLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y: torch.Tensor, y_teacher: torch.Tensor, mask: torch.Tensor):
        cons_loss = ((y - y_teacher) ** 2).sum(dim=1)
        cons_loss = cons_loss * mask
        if self.reduction == 'mean':
            return cons_loss.mean()
        else:
            return cons_loss


class ClassBalanceLoss(nn.Module):
    """Class balance loss
    """
    def __init__(self, num_classes):
        super(ClassBalanceLoss, self).__init__()
        self.uniform_distribution = torch.ones(num_classes) / num_classes

    def forward(self, y: torch.Tensor):
        return F.binary_cross_entropy(y.mean(dim=0), self.uniform_distribution.to(y.device))


class EmaTeacher(object):
    r"""Exponential moving average model
    Examples::

        >>> #initialize classifier
        >>> classifier = ImageClassifier()
        >>> teacher = EmaTeacher(classifier, 0.9)
        >>> num_iterations = 10000
        >>> for _ in range(num_iterations):
        >>>     # compute teacher output
        >>>     x = torch.randn(32, 31)
        >>>     y_teacher = teacher(x)
        >>>     # update teacher with current classifier (eg. after optimizer.step())
        >>>     teacher.update()
    """

    def __init__(self, model, alpha):
        self.model = model
        self.alpha = alpha
        self.teacher = copy.deepcopy(model)
        set_requires_grad(self.teacher, False)

    def update(self):
        """Perform ema update
        """
        for teacher_param, param in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_param.data = self.alpha * teacher_param + (1 - self.alpha) * param

    def __call__(self, x: torch.Tensor):
        return self.teacher(x)

    def train(self, mode: Optional[bool] = True):
        self.teacher.train(mode)


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
