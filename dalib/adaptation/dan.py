import torch
import torch.nn as nn
from dalib.modules.classifier import Classifier as ClassifierBase

__all__ = ['MultipleKernelMaximumMeanDiscrepancy', 'GaussianKernel']


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks <https://arxiv.org/pdf/1502.02791>`_

    .. TODO math definitions

    Parameters:
        - kernels (list of class:`nn.Module` object): multiple kernels to compute the domain discrepancy

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Output: scalar

    Examples::
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = [GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.)]
        >>> loss = adaptation.dan.MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(f_s, f_t)
    """
    def __init__(self, kernels):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        if not isinstance(kernels, list):
            self.kernels = [kernels]
        else:
            self.kernels = kernels
        self.index_matrix = None

    def forward(self, f_s, f_t):
        features = torch.cat([f_s, f_t], dim=0)
        batch_size = int(f_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix)
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])
        loss = (kernel_matrix * self.index_matrix).sum() / float(batch_size)
        return loss


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    The running estimates are kept with a default :attr:`momentum` of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\sigma^2_\text{new} = (1 - \text{momentum}) \times \sigma^2 + \text{momentum} \times \sigma^2_t`,
        where :math:`\sigma^2_\text{new}` is the estimated statistic and
        :math:`\sigma^2_t = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2` is the
        new observed value.

    Parameters:
        - sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        - momentum (float, optional): value used for the :math:`\sigma^2` computation. Default: 0.1
        - track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        - alpha (float, optional): decide the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Shape:
        - Input: :math:`(N, F)` where F means the dimension of input features.
        - Output: :math:`(N, N)`
    """
    def __init__(self, sigma=None, momentum=0.1, track_running_stats=True, alpha=1.):
        super(GaussianKernel, self).__init__()
        self.sigma_square = torch.tensor(sigma * sigma).cuda() if sigma is not None else None
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X):
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.training and self.track_running_stats:
            if self.sigma_square is None:
                self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())
            else:
                self.sigma_square = (1. - self.momentum) * self.sigma_square \
                                    + self.momentum * self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


# TODO JointMaximumMeanDiscrepancy has not achieved high accuracy yet.
class JointMaximumMeanDiscrepancy(nn.Module):
    def __init__(self, kernels):
        super(JointMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None

    def forward(self, f_s, f_t):
        batch_size = int(f_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix)

        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_f_s, layer_f_t, layer_kernels in zip(f_s, f_t, self.kernels):
            layer_features = torch.cat([layer_f_s, layer_f_t], dim=0)
            kernel_matrix *= sum([kernel(layer_features) for kernel in layer_kernels])

        loss = (kernel_matrix * self.index_matrix).sum() / float(batch_size)
        return loss


def _update_index_matrix(batch_size, index_matrix=None):
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    E.g. when batch_size = 3, index_matrix is
            [[ 0.,  1.,  0.,  0., -1., -1.],
            [ 0.,  0.,  1., -1.,  0., -1.],
            [ 1.,  0.,  0., -1., -1.,  0.],
            [ 0.,  0.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1.],
            [ 0.,  0.,  0.,  1.,  0.,  0.]]
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size).cuda()
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            index_matrix[s1, s2] = 1.
            index_matrix[t1, t2] = 1.
            index_matrix[s1, t2] = -1.
            index_matrix[s2, t1] = -1.
    return index_matrix


class Classifier(ClassifierBase):
    def __init__(self, backbone, num_classes, bottleneck_dim=256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        super(Classifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

