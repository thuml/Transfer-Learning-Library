import copy
from typing import Optional
import torch


def set_requires_grad(net, requires_grad=False):
    """
    Set requires_grad=False for all the parameters to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


class EMATeacher(object):
    r"""
    Exponential moving average model from `Mean teachers are better role models: Weight-averaged consistency targets
    improve semi-supervised deep learning results (NIPS 2017) <https://arxiv.org/abs/1703.01780>`_

    We use :math:`\theta_t'` to denote parameters of the teacher model at training step t, use :math:`\theta_t` to
    denote parameters of the student model at training step t. Given decay factor :math:`\alpha`,
    we update the teacher model in an exponential moving average manner

    .. math::
        \theta_t'=\alpha \theta_{t-1}' + (1-\alpha)\theta_t

    Args:
        model (torch.nn.Module): the student model
        alpha (float): decay factor for EMA.

    Inputs:
        x (tensor): input tensor

    Examples::

        >>> classifier = ImageClassifier(backbone, num_classes=31, bottleneck_dim=256).to(device)
        >>> # initialize teacher model
        >>> teacher = EMATeacher(classifier, 0.9)
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


def update_bn(model, ema_model):
    """
    Replace batch normalization statistics of the teacher model with that ot the student model
    """
    for m2, m1 in zip(ema_model.named_modules(), model.named_modules()):
        if ('bn' in m2[0]) and ('bn' in m1[0]):
            bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
            bn2['running_mean'].data.copy_(bn1['running_mean'].data)
            bn2['running_var'].data.copy_(bn1['running_var'].data)
            bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)
