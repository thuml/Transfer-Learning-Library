"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Optional
import torch
import torch.nn as nn
from tllib.modules.classifier import Classifier as ClassifierBase


class BatchSpectralPenalizationLoss(nn.Module):
    r"""Batch spectral penalization loss from `Transferability vs. Discriminability: Batch
    Spectral Penalization for Adversarial Domain Adaptation (ICML 2019)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/batch-spectral-penalization-icml19.pdf>`_.

    Given source features :math:`f_s` and target features :math:`f_t` in current mini batch, singular value
    decomposition is first performed

    .. math::
        f_s = U_s\Sigma_sV_s^T

    .. math::
        f_t = U_t\Sigma_tV_t^T

    Then batch spectral penalization loss is calculated as

    .. math::
        loss=\sum_{i=1}^k(\sigma_{s,i}^2+\sigma_{t,i}^2)

    where :math:`\sigma_{s,i},\sigma_{t,i}` refer to the :math:`i-th` largest singular value of source features
    and target features respectively. We empirically set :math:`k=1`.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    """

    def __init__(self):
        super(BatchSpectralPenalizationLoss, self).__init__()

    def forward(self, f_s, f_t):
        _, s_s, _ = torch.svd(f_s)
        _, s_t, _ = torch.svd(f_t)
        loss = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return loss


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
