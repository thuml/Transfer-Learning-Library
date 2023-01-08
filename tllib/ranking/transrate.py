"""
@author: Louis Fouquet
@contact: louisfouquet75@gmail.com
"""
import numpy as np

__all__ = ['transrate']


def coding_rate(features: np.ndarray, eps=1e-4):
    f = features
    n, d = f.shape
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * f.transpose() @ f))
    return 0.5 * rate


def transrate(features: np.ndarray, labels: np.ndarray, eps=1e-4):
    r"""
    TransRate in `Frustratingly easy transferability estimation (ICML 2022) 
    <https://proceedings.mlr.press/v162/huang22d/huang22d.pdf>`_.
    
    The TransRate :math:`TrR` can be described as:

    .. math::
        TrR= R\left(f, \espilon \right) - R\left(f, \espilon \mid y \right) 
    
    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector, 
    :math:`R` is the coding rate with distortion rate :math:`\epsilon`

    Args:
        features (np.ndarray):features extracted by pre-trained model.
        labels (np.ndarray):  groud-truth labels.
        eps (float, optional): distortion rare (Default: 1e-4)

    Shape:
        - features: (N, F), with number of samples N and feature dimension F.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar.
    """
    f = features
    y = labels
    f = f - np.mean(f, axis=0, keepdims=True)
    Rf = coding_rate(f, eps)
    Rfy = 0.0
    C = int(y.max() + 1)
    for i in range(C):
        Rfy += coding_rate(f[(y == i).flatten()], eps)
    return Rf - Rfy / C
