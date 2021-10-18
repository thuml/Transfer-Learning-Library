"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Optional, Callable
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.modules.classifier import Classifier as ClassifierBase
from dalib.translation.cyclegan.util import set_requires_grad


class ConsistencyLoss(nn.Module):
    r"""
    Consistency loss between output of student model and output of teacher model.
    Given distance measure :math:`D`, student model's output :math:`y`, teacher
    model's output :math:`y_{teacher}`, binary mask :math:`mask`, consistency loss is

    .. math::
        D(y, y_{teacher}) * mask

    Args:
        distance_measure (callable): Distance measure function.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - y: predictions from student model
        - y_teacher: predictions from teacher model
        - mask: binary mask

    Shape:
        - y, y_teacher: :math:`(N, C)` where C means the number of classes.
        - mask: :math:`(N, )` where N means mini-batch size.
    """

    def __init__(self, distance_measure: Callable, reduction: Optional[str] = 'mean'):
        super(ConsistencyLoss, self).__init__()
        self.distance_measure = distance_measure
        self.reduction = reduction

    def forward(self, y: torch.Tensor, y_teacher: torch.Tensor, mask: torch.Tensor):
        cons_loss = self.distance_measure(y, y_teacher)
        cons_loss = cons_loss * mask
        if self.reduction == 'mean':
            return cons_loss.mean()
        else:
            return cons_loss


class L2ConsistencyLoss(ConsistencyLoss):
    r"""
    L2 consistency loss. Given student model's output :math:`y`, teacher model's output :math:`y_{teacher}`
    and binary mask :math:`mask`, L2 consistency loss is

    .. math::
        \text{MSELoss}(y, y_{teacher}) * mask

    """

    def __init__(self, reduction: Optional[str] = 'mean'):
        def l2_distance(y: torch.Tensor, y_teacher: torch.Tensor):
            return ((y - y_teacher) ** 2).sum(dim=1)

        super(L2ConsistencyLoss, self).__init__(l2_distance, reduction)


class ClassBalanceLoss(nn.Module):
    r"""
    Class balance loss that penalises the network for making predictions that exhibit large class imbalance.
    Given predictions :math:`y` with dimension :math:`(N, C)`, we first calculate mean across mini-batch dimension,
    resulting in mini-batch mean per-class probability :math:`y_{mean}` with dimension :math:`(C, )`

    .. math::
        y_{mean}^j = \frac{1}{N} \sum_{i=1}^N y_i^j

    Then we calculate binary cross entropy loss between :math:`y_{mean}` and uniform probability vector :math:`u` with
    the same dimension where :math:`u^j` = :math:`\frac{1}{C}`

    .. math::
        loss = \text{BCELoss}(y_{mean}, u)

    Args:
        num_classes (int): Number of classes

    Inputs:
        - y (tensor): predictions from classifier

    Shape:
        - y: :math:`(N, C)` where C means the number of classes.
    """

    def __init__(self, num_classes):
        super(ClassBalanceLoss, self).__init__()
        self.uniform_distribution = torch.ones(num_classes) / num_classes

    def forward(self, y: torch.Tensor):
        return F.binary_cross_entropy(y.mean(dim=0), self.uniform_distribution.to(y.device))


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
