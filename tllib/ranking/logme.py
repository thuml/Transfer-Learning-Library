"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""
import os
import torch
import numpy as np
from numba import njit
import torch.nn as nn
from typing import List, Dict

__all__ = ['log_maximum_evidence', 'PosteriorPredictiveAlignment', 'FeatureAlignment']


class FeatureAlignment(nn.Module):
    """
        f_t: teachers features
        f_s: student features
    """
    def __init__(self, student: nn.Module, teachers: List[nn.Module]):
        super(FeatureAlignment, self).__init__()
        self.criterion = nn.MSELoss()
        self.transforms = nn.ModuleList(
            [nn.Linear(student.features_dim, teacher.features_dim) for teacher in teachers]
        )

    def forward(self, f_t, f_s):
        alignment_loss = 0.0
        for i, f in enumerate(f_t):
            f_e = self.transforms[i](f_s)
            alignment_loss += self.criterion(f, f_e)
        alignment_loss /= len(self.transforms)

        return alignment_loss

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        print(self.transforms)
        print(base_lr)
        print(self.transforms.parameters())
        params = [
            {"params": self.transforms.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


class PosteriorPredictiveAlignment(nn.Module):
    """
    Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models
    for Transfer Learning (ICML 2021) <https://arxiv.org/pdf/2102.11005.pdf>`
    
    Args:
        f_t: teachers features
        f_s: student features
    """
    def __init__(self, path: str, student: str, teachers: List[str], temperature: float, device):
        super(PosteriorPredictiveAlignment, self).__init__()
        self.criterion = nn.MSELoss()
        self.student_dist = torch.from_numpy(
                np.load(os.path.join(path, f'{student}/distribution.npz'))['distribution']
            ).float().to(device)
            
        self.teachers_dists = []
        teachers_logmes = []
        for teacher in teachers:
            npz_file = np.load(os.path.join(path, f'{teacher}/distribution.npz'))
            teacher_dist = torch.from_numpy(npz_file['distribution']).float().to(device)
            self.teachers_dists.append(teacher_dist)
            teachers_logmes.append(npz_file['logme'].item())
        self.weights = torch.softmax(torch.tensor(teachers_logmes) / temperature, dim=0)
    

    def forward(self, f_t, f_s):
        f_s = torch.matmul(f_s, self.student_dist.t())
        f_e = torch.zeros_like(f_s)

        for i, f in enumerate(f_t):
            f_e += self.weights[i] * torch.matmul(f, self.teachers_dists[i].t()) # B x C
        alignment_loss = self.criterion(f_s, f_e)

        return alignment_loss


def log_maximum_evidence(features: np.ndarray, targets: np.ndarray, regression=False, return_distribution=False):
    """
    Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models
    for Transfer Learning (ICML 2021) <https://arxiv.org/pdf/2102.11005.pdf>`
    
    Args:
        - features (np.ndarray): feature matrix from pre-trained model.
        - targets (np.ndarray): targets labels/values.
        - regression: whether to apply in regression setting.
        - return_weights: whether to return bayesian weight.

    Shape:
        - features: [N, F] with element in [0, C_t)  and feature dimension F.
        - targets: [N] or [N, C], with C regression-labels.
        - weights: [F, C_t].
        - score: scaler.
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
    distribution = []
    if regression:
        C = y.shape[1]
        for i in range(C):
            y_ = y[:, i]
            evidence, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            distribution.append(m)
    else:
        C = int(y.max() + 1)
        for i in range(C):
            y_ = (y == i).astype(np.float64)
            evidence, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            distribution.append(m)
            
    score = np.mean(evidences)
    distribution = np.vstack(distribution)

    if return_distribution:
        return score, distribution
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
