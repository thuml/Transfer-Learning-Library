"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""
import numpy as np
from numba import njit

__all__ = ['log_maximum_evidence']


def log_maximum_evidence(features: np.ndarray, targets: np.ndarray, regression=False, return_weights=False):
    r"""
    Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models
    for Transfer Learning (ICML 2021) <https://arxiv.org/pdf/2102.11005.pdf>`_.
    
    Args:
        features (np.ndarray): feature matrix from pre-trained model.
        targets (np.ndarray): targets labels/values.
        regression (bool, optional): whether to apply in regression setting. (Default: False)
        return_weights (bool, optional): whether to return bayesian weight. (Default: False)

    Shape:
        - features: (N, F) with element in [0, :math:`C_t`) and feature dimension F, where :math:`C_t` denotes the number of target class
        - targets: (N, ) or (N, C), with C regression-labels.
        - weights: (F, :math:`C_t`).
        - score: scalar.
    """
    f = features.astype(np.float64)
    y = targets
    if regression:
        y = targets.astype(np.float64)

    fh = f
    f = f.transpose()
    D, N = f.shape
    v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

    evidences = []
    weights = []
    if regression:
        C = y.shape[1]
        for i in range(C):
            y_ = y[:, i]
            evidence, weight = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            weights.append(weight)
    else:
        C = int(y.max() + 1)
        for i in range(C):
            y_ = (y == i).astype(np.float64)
            evidence, weight = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            weights.append(weight)

    score = np.mean(evidences)
    weights = np.vstack(weights)

    if return_weights:
        return score, weights
    else:
        return score


@njit
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ y_))

    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / alpha_de
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / beta_de
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam

    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * beta_de \
               - alpha / 2.0 * alpha_de \
               - N / 2.0 * np.log(2 * np.pi)

    return evidence / N, m
