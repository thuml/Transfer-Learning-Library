"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
import torch
import torch.nn as nn

__all__ = ['BatchSpectralShrinkage']


class BatchSpectralShrinkage(nn.Module):
    r"""
    The regularization term in `Catastrophic Forgetting Meets Negative Transfer:
    Batch Spectral Shrinkage for Safe Transfer Learning (NIPS 2019) <https://proceedings.neurips.cc/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf>`_.


    The BSS regularization of feature matrix :math:`F` can be described as:

    .. math::
        L_{bss}(F) = \sum_{i=1}^{k} \sigma_{-i}^2 ,

    where :math:`k` is the number of singular values to be penalized, :math:`\sigma_{-i}` is the :math:`i`-th smallest singular value of feature matrix :math:`F`.

    All the singular values of feature matrix :math:`F` are computed by `SVD`:

    .. math::
        F = U\Sigma V^T,

    where the main diagonal elements of the singular value matrix :math:`\Sigma` is :math:`[\sigma_1, \sigma_2, ..., \sigma_b]`.


    Args:
        k (int):  The number of singular values to be penalized. Default: 1

    Shape:
        - Input: :math:`(b, |\mathcal{f}|)` where :math:`b` is the batch size and :math:`|\mathcal{f}|` is feature dimension.
        - Output: scalar.

    """
    def __init__(self, k=1):
        super(BatchSpectralShrinkage, self).__init__()
        self.k = k

    def forward(self, feature):
        result = 0
        u, s, v = torch.svd(feature.t())
        num = s.size(0)
        for i in range(self.k):
            result += torch.pow(s[num-1-i], 2)
        return result
