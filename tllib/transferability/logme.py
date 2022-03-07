"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""
import numpy as np


__all__ = ['LogME']

def LogME(source_feature: np.ndarray, target: np.ndarray, regression=False, return_weight=False):
    """
    Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models
     for Transfer Learning (ICML 2021) <https://arxiv.org/pdf/2102.11005.pdf>`

    :param source_feature: [N, F], feature matrix from pre-trained model
    :param target: target labels/values
        For classification, y has shape [N] with element in [0, C_t).
        For regression, y has shape [N, C] with C regression-labels
    :param regression: whether regression
    :return: LogME score
    """
    f = source_feature.astype(np.float64)
    y = target
    if regression:
        y = target.astype(np.float64)

    fh = f
    f = f.transpose()
    D, N = f.shape
    v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

    evidences = []
    weights = []
    if regression:
        K = y.shape[1]
        for i in range(K):
            y_ = y[:, i]
            evidence, weight = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            weights.append(weight)
    else:
        K = int(y.max() + 1)
        for i in range(K):
            y_ = (y == i).astype(np.float64)
            evidence, weight = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            weights.append(weight)

    return np.mean(evidences), np.vstack(weights)


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
