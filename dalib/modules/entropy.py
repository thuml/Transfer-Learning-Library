"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch


def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H
