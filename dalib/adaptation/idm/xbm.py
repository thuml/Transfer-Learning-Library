"""
Modified from https://github.com/SikaStar/IDM
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch


class XBM(object):
    """`XBM` module in `Cross-Batch Memory for Embedding Learning (CVPR 2020) <https://arxiv.org/pdf/1912.06798.pdf>`_.
    """
    def __init__(self, memory_size, feature_size):
        self.K = memory_size
        self.D = feature_size
        self.feats = torch.zeros(self.K, self.D).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != 0

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size

    def clean_target_domain(self, source_classes, target_classes):
        empty_feats = torch.zeros(self.K, self.D).cuda()
        empty_targets = torch.zeros(self.K, dtype=torch.long).cuda()
        j = 0
        for i in range(self.K):
            if source_classes <= self.targets[i] < source_classes + target_classes:
                continue
            else:
                empty_feats[j] = self.feats[i]
                empty_targets[j] = self.targets[i]
                j += 1
        self.feats = empty_feats
        self.targets = empty_targets
        if j > 0:
            self.ptr = j
