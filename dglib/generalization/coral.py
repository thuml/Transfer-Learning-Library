import torch
import torch.nn as nn


class CorrelationAlignmentLoss(nn.Module):

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_i: torch.Tensor, f_j: torch.Tensor) -> torch.Tensor:
        mean_i = f_i.mean(0, keepdim=True)
        mean_j = f_j.mean(0, keepdim=True)
        cent_i = f_i - mean_i
        cent_j = f_j - mean_j
        cov_i = torch.mm(cent_i.t(), cent_i) / (len(f_i) - 1)
        cov_j = torch.mm(cent_j.t(), cent_j) / (len(f_j) - 1)

        mean_diff = (mean_i - mean_j).pow(2).mean()
        cov_diff = (cov_i - cov_j).pow(2).mean()

        return mean_diff + cov_diff
