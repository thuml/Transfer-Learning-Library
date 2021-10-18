"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn.functional as F
import math


def robust_entropy(y, ita=1.5, num_classes=19, reduction='mean'):
    """ Robust entropy proposed in `FDA: Fourier Domain Adaptation for Semantic Segmentation (CVPR 2020) <https://arxiv.org/abs/2004.05498>`_

    Args:
        y (tensor): logits output of segmentation model in shape of :math:`(N, C, H, W)`
        ita (float, optional): parameters for robust entropy. Default: 1.5
        num_classes (int, optional): number of classes. Default: 19
        reduction (string, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Returns:
        Scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    """
    P = F.softmax(y, dim=1)
    logP = F.log_softmax(y, dim=1)
    PlogP = P * logP
    ent = -1.0 * PlogP.sum(dim=1)
    ent = ent / math.log(num_classes)

    # compute robust entropy
    ent = ent ** 2.0 + 1e-8
    ent = ent ** ita

    if reduction == 'mean':
        return ent.mean()
    else:
        return ent
