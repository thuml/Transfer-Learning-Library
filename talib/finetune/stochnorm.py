"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

__all__ = ['StochNorm1d', 'StochNorm2d', 'convert_model']


class _StochNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, p=0.5):
        super(_StochNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.p = p
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        if self.training:
            z_0 = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                False, self.momentum, self.eps)

            z_1 = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                True, self.momentum, self.eps)

            if input.dim() == 2:
                s = torch.from_numpy(
                    np.random.binomial(n=1, p=self.p, size=self.num_features).reshape(1,
                                                                                      self.num_features)).float().cuda()
            elif input.dim() == 3:
                s = torch.from_numpy(
                    np.random.binomial(n=1, p=self.p, size=self.num_features).reshape(1, self.num_features,
                                                                                      1)).float().cuda()
            elif input.dim() == 4:
                s = torch.from_numpy(
                    np.random.binomial(n=1, p=self.p, size=self.num_features).reshape(1, self.num_features, 1,
                                                                                      1)).float().cuda()
            else:
                raise BaseException()

            z = (1 - s) * z_0 + s * z_1
        else:
            z = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                False, self.momentum, self.eps)

        return z


class StochNorm1d(_StochNorm):
    r"""Applies Stochastic Normalization over a 2D or 3D input (a mini-batch of 1D inputs with optional additional channel dimension)

    Stochastic  Normalization is proposed in `Stochastic Normalization (NIPS 2020) <https://papers.nips.cc/paper/2020/file/bc573864331a9e42e4511de6f678aa83-Paper.pdf>`_

    .. math::

        \hat{x}_{i,0} = \frac{x_i - \tilde{\mu}}{ \sqrt{\tilde{\sigma} + \epsilon}}

        \hat{x}_{i,1} = \frac{x_i - \mu}{ \sqrt{\sigma + \epsilon}}

        \hat{x}_i = (1-s)\cdot \hat{x}_{i,0} + s\cdot \hat{x}_{i,1}

         y_i = \gamma \hat{x}_i + \beta

    where :math:`\mu` and :math:`\sigma` are mean and variance of current mini-batch data.

    :math:`\tilde{\mu}` and :math:`\tilde{\sigma}` are current moving statistics of training data.

    :math:`s` is a branch-selection variable generated from a Bernoulli distribution, where :math:`P(s=1)=p`.


    During training, there are two normalization branches. One uses mean and
    variance of current mini-batch data, while the other uses current moving
    statistics of the training data as usual batch normalization.

    During evaluation, the moving statistics is used for normalization.


    Args:
        num_features (int): :math:`c` from an expected input of size :math:`(b, c, l)` or  :math:`l` from an expected input of size :math:`(b, l)`.
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-5
        momentum (float): The value used for the running_mean and running_var
            computation. Default: 0.1
        affine (bool): A boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
        track_running_stats (bool): A boolean value that when set to True, this module tracks
         the running mean and variance, and when set to False, this module does not
         track such statistics, and initializes statistics buffers running_mean and
         running_var as None. When these buffers are None, this module always uses
         batch statistics in both training and eval modes. Default: True
         p (float): The probability to choose the second branch (usual BN). Default: 0.5

    Shape:
        - Input: :math:`(b, l)` or :math:`(b, c, l)`
        - Output: :math:`(b, l)` or :math:`(b, c, l)` (same shape as input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class StochNorm2d(_StochNorm):
    r"""
    Applies Stochastic  Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)

    Stochastic  Normalization is proposed in `Stochastic Normalization (NIPS 2020) <https://papers.nips.cc/paper/2020/file/bc573864331a9e42e4511de6f678aa83-Paper.pdf>`_

    .. math::

        \hat{x}_{i,0} = \frac{x_i - \tilde{\mu}}{ \sqrt{\tilde{\sigma} + \epsilon}}

        \hat{x}_{i,1} = \frac{x_i - \mu}{ \sqrt{\sigma + \epsilon}}

        \hat{x}_i = (1-s)\cdot \hat{x}_{i,0} + s\cdot \hat{x}_{i,1}

         y_i = \gamma \hat{x}_i + \beta

    where :math:`\mu` and :math:`\sigma` are mean and variance of current mini-batch data.

    :math:`\tilde{\mu}` and :math:`\tilde{\sigma}` are current moving statistics of training data.

    :math:`s` is a branch-selection variable generated from a Bernoulli distribution, where :math:`P(s=1)=p`.


    During training, there are two normalization branches. One uses mean and
    variance of current mini-batch data, while the other uses current moving
    statistics of the training data as usual batch normalization.

    During evaluation, the moving statistics is used for normalization.


    Args:
        num_features (int): :math:`c` from an expected input of size :math:`(b, c, h, w)`.
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-5
        momentum (float): The value used for the running_mean and running_var
            computation. Default: 0.1
        affine (bool): A boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
        track_running_stats (bool): A boolean value that when set to True, this module tracks
         the running mean and variance, and when set to False, this module does not
         track such statistics, and initializes statistics buffers running_mean and
         running_var as None. When these buffers are None, this module always uses
         batch statistics in both training and eval modes. Default: True
         p (float): The probability to choose the second branch (usual BN). Default: 0.5

    Shape:
        - Input: :math:`(b, c, h, w)`
        - Output: :math:`(b, c, h, w)` (same shape as input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class StochNorm3d(_StochNorm):
    r"""
    Applies Stochastic  Normalization over a 5D input (a mini-batch of 3D inputs with additional channel dimension)

    Stochastic  Normalization is proposed in `Stochastic Normalization (NIPS 2020) <https://papers.nips.cc/paper/2020/file/bc573864331a9e42e4511de6f678aa83-Paper.pdf>`_

    .. math::

        \hat{x}_{i,0} = \frac{x_i - \tilde{\mu}}{ \sqrt{\tilde{\sigma} + \epsilon}}

        \hat{x}_{i,1} = \frac{x_i - \mu}{ \sqrt{\sigma + \epsilon}}

        \hat{x}_i = (1-s)\cdot \hat{x}_{i,0} + s\cdot \hat{x}_{i,1}

         y_i = \gamma \hat{x}_i + \beta

    where :math:`\mu` and :math:`\sigma` are mean and variance of current mini-batch data.

    :math:`\tilde{\mu}` and :math:`\tilde{\sigma}` are current moving statistics of training data.

    :math:`s` is a branch-selection variable generated from a Bernoulli distribution, where :math:`P(s=1)=p`.


    During training, there are two normalization branches. One uses mean and
    variance of current mini-batch data, while the other uses current moving
    statistics of the training data as usual batch normalization.

    During evaluation, the moving statistics is used for normalization.


    Args:
        num_features (int): :math:`c` from an expected input of size :math:`(b, c, d, h, w)`
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-5
        momentum (float): The value used for the running_mean and running_var
            computation. Default: 0.1
        affine (bool): A boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
        track_running_stats (bool): A boolean value that when set to True, this module tracks
         the running mean and variance, and when set to False, this module does not
         track such statistics, and initializes statistics buffers running_mean and
         running_var as None. When these buffers are None, this module always uses
         batch statistics in both training and eval modes. Default: True
         p (float): The probability to choose the second branch (usual BN). Default: 0.5

    Shape:
        - Input: :math:`(b, c, d, h, w)`
        - Output: :math:`(b, c, d, h, w)` (same shape as input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


def convert_model(module, p):
    """
    Traverses the input module and its child recursively and replaces all
    instance of BatchNorm to StochNorm.

    Args:
        module (torch.nn.Module): The input module needs to be convert to StochNorm model.
        p (float): The hyper-parameter for StochNorm layer.

    Returns:
         The module converted to StochNorm version.
    """

    mod = module
    for pth_module, stoch_module in zip([torch.nn.modules.batchnorm.BatchNorm1d,
                                         torch.nn.modules.batchnorm.BatchNorm2d,
                                         torch.nn.modules.batchnorm.BatchNorm3d],
                                        [StochNorm1d,
                                         StochNorm2d,
                                         StochNorm3d]):
        if isinstance(module, pth_module):
            mod = stoch_module(module.num_features, module.eps, module.momentum, module.affine, p)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var

            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child, p))

    return mod
