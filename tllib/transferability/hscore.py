"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""
import numpy as np


__all__ = ['HScore']

def HScore(source_feature: np.ndarray, target_label: np.ndarray):
    """
    H-score in `An Information-theoretic Approach to Transferability in Task Transfer Learning (ICIP 2019) 
    <http://yangli-feasibility.com/home/media/icip-19.pdf>`
    :param source_feature: [N, F], feature matrix from pre-trained model
    :param target_label: [N], target label, elements in [0, C_t)
    :return: H-score
    """
    f = source_feature
    y = target_label

    def covariance(X):
        X_mean = X - np.mean(X, axis=0, keepdims=True)
        cov = np.divide(np.dot(X_mean.T, X_mean), len(X) - 1)
        return cov
        
    covf = covariance(f)
    covg = covariance(g)

    K = int(y.max() + 1)
    g = np.zeros_like(f)

    for i in K:
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))

    return score



