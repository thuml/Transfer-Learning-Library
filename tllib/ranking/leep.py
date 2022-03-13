"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""

import numpy as np


__all__ = ['log_expected_empirical_prediction']

def log_expected_empirical_prediction(source_pred: np.ndarray, target_label: np.ndarray):
    """
    Log Expected Empirical Prediction in 'LEEP: A New Measure to 
    Evaluate Transferability of Learned Representations (ICML 2020)
    <http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf>'

    :param source_pred: shape [N, C_s] source data on pre-trained model's predictions
    :param target_label: shape [N], elements in [0, C_t)
    :return: LEEP score
    """
    N, C_s = source_pred.shape
    target_label = target_label.reshape(-1)
    C_t = int(np.max(target_label) + 1)         # the number of target classes
    normalized_prob = source_pred / float(N)   # sum(normalized_prob) = 1
    joint = np.zeros((C_t, C_s), dtype=float)   # placeholder for joint distribution over (y, z)
    
    for i in range(C_t):
        this_class = normalized_prob[target_label == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row

    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    empirical_prediction = source_pred @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, target_label)])
    leep_score = np.mean(np.log(empirical_prob))
    
    return leep_score