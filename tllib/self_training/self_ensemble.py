"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from tllib.modules.classifier import Classifier as ClassifierBase


class ClassBalanceLoss(nn.Module):
    r"""
    Class balance loss that penalises the network for making predictions that exhibit large class imbalance.
    Given predictions :math:`p` with dimension :math:`(N, C)`, we first calculate
    the mini-batch mean per-class probability :math:`p_{mean}` with dimension :math:`(C, )`, where

    .. math::
        p_{mean}^j = \frac{1}{N} \sum_{i=1}^N p_i^j

    Then we calculate binary cross entropy loss between :math:`p_{mean}` and uniform probability vector :math:`u` with
    the same dimension where :math:`u^j` = :math:`\frac{1}{C}`

    .. math::
        loss = \text{BCELoss}(p_{mean}, u)

    Args:
        num_classes (int): Number of classes

    Inputs:
        - p (tensor): predictions from classifier

    Shape:
        - p: :math:`(N, C)` where C means the number of classes.
    """

    def __init__(self, num_classes):
        super(ClassBalanceLoss, self).__init__()
        self.uniform_distribution = torch.ones(num_classes) / num_classes

    def forward(self, p: torch.Tensor):
        return F.binary_cross_entropy(p.mean(dim=0), self.uniform_distribution.to(p.device))


class EmaTeacher(object):
    r"""
    Exponential moving average model used in `Self-ensembling for Visual Domain Adaptation (ICLR 2018) <https://arxiv.org/abs/1706.05208>`_

    We denote :math:`\theta_t'` as the parameters of teacher model at training step t, :math:`\theta_t` as the
    parameters of student model at training step t, :math:`\alpha` as decay rate. Then we update teacher model in an
    exponential moving average manner as follows

    .. math::
        \theta_t'=\alpha \theta_{t-1}' + (1-\alpha)\theta_t

    Args:
        model (torch.nn.Module): student model
        alpha (float): decay rate for EMA.

    Inputs:
        x (tensor): input data fed to teacher model

    Examples::

        >>> classifier = ImageClassifier(backbone, num_classes=31, bottleneck_dim=256).to(device)
        >>> # initialize teacher model
        >>> teacher = EmaTeacher(classifier, 0.9)
        >>> num_iterations = 1000
        >>> for _ in range(num_iterations):
        >>>     # x denotes input of one mini-batch
        >>>     # you can get teacher model's output by teacher(x)
        >>>     y_teacher = teacher(x)
        >>>     # when you want to update teacher, you should call teacher.update()
        >>>     teacher.update()
    """

    def __init__(self, model, alpha):
        self.model = model
        self.alpha = alpha
        self.teacher = copy.deepcopy(model)
        set_requires_grad(self.teacher, False)

    def set_alpha(self, alpha: float):
        assert alpha >= 0
        self.alpha = alpha

    def update(self):
        for teacher_param, param in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_param.data = self.alpha * teacher_param + (1 - self.alpha) * param

    def __call__(self, x: torch.Tensor):
        return self.teacher(x)

    def train(self, mode: Optional[bool] = True):
        self.teacher.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.teacher.state_dict()

    def load_state_dict(self, state_dict):
        self.teacher.load_state_dict(state_dict)

    @property
    def module(self):
        return self.teacher.module


def set_requires_grad(net, requires_grad=False):
    """
    Set requires_grad=False for all the parameters to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
