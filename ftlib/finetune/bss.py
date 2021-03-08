import torch
import torch.nn as nn


class BatchSpectralShrinkage(nn.Module):
    def __init__(self, k=1):
        super(BatchSpectralShrinkage, self).__init__()
        self.k = k

    def forward(self, feature):
        result = 0
        u, s, v = torch.svd(feature.t())
        num = s.size(0)
        for i in range(self.k):
            result += torch.pow(s[num-1-i], 2)
        return result
