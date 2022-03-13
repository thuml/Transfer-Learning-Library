"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""
import numpy as np


__all__ = ['negative_conditional_entropy']

def negative_conditional_entropy(source_label: np.ndarray, target_label: np.ndarray):
    """
    Negative Conditional Entropy in `Transferability and Hardness of Supervised 
    Classification Tasks (ICCV 2019) <https://arxiv.org/pdf/1908.08142v1.pdf>`
    
    :param source_label: shape [N], elements in [0, C_s), often got from taken argmax from pre-trained predictions
    :param target_label: shape [N], elements in [0, C_t)
    :return: NCE score 
    """
    C_t = int(np.max(target_label) + 1)  # the number of target classes
    C_s = int(np.max(source_label) + 1)  # the number of source classes
    N = len(source_label)

    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for the joint distribution, shape [C_t, C_s]
    for s, t in zip(source_label, target_label):
        s = int(s)
        t = int(t)
        joint[t, s] += 1.0 / N
    p_z = joint.sum(axis=0, keepdims=True)  # shape [1, C_s]
    print('p_z', p_z)
    
    p_target_given_source = (joint / p_z).T  # P(y | z), shape [C_s, C_t]
    mask = p_z.reshape(-1) != 0  # valid Z, shape [C_s]
    p_target_given_source = p_target_given_source[mask] + 1e-20  # remove NaN where p(z) = 0, add 1e-20 to avoid log (0)
    entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1, keepdims=True)  # shape [C_s, 1]
    conditional_entropy = np.sum(entropy_y_given_z * p_z.reshape((-1, 1))[mask]) # scalar
    
    return - conditional_entropy
