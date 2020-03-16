import torch
import torch.nn as nn

__all__ = ['GaussianKernel']


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
        - alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
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


