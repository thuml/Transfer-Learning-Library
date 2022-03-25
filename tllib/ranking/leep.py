"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""

import numpy as np


__all__ = ['log_expected_empirical_prediction']

def log_expected_empirical_prediction(predictons: np.ndarray, labels: np.ndarray):
    """
    Log Expected Empirical Prediction in 'LEEP: A New Measure to 
    Evaluate Transferability of Learned Representations (ICML 2020)
    <http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf>'
    
    The LEEP :math:`\mathcal{T}` can be described as:

    .. math::
        \mathcal{T}=\mathbb{E}\log \left(\sum_{z \in \mathcal{C}_s} \hat{P}\left(y \mid z\right) \theta\left(y \right)_{z}\right)

    where :math `\theta\left(y\right)_{z}` is the predictions of pre-trained model on source category, :math:`\hat{P}\left(y \mid z\right)` is the empirical conditional distribution estimated by prediction and ground-truth label.

    Args:
        - predictons (np.ndarray): predictions of pre-trained model.
        - labels (np.ndarray): groud-truth labels.

    Shape: 
        - predictons: [N, C_s], with number of samples N and source class number C_s.
        - labels: [N] elements in [0, C_t), with target class number C_t.
        - score: scaler
    """
    N, C_s = predictons.shape
    labels = labels.reshape(-1)
    C_t = int(np.max(labels) + 1)   

    normalized_prob = predictons / float(N)  
    joint = np.zeros((C_t, C_s), dtype=float)   # placeholder for joint distribution over (y, z)
    
    for i in range(C_t):
        this_class = normalized_prob[labels == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row

    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    empirical_prediction = predictons @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, labels)])
    score = np.mean(np.log(empirical_prob))
    
    return score