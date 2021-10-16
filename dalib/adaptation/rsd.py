"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn
import torch


class RepresentationSubspaceDistance(nn.Module):
    """
    `Representation Subspace Distance (ICML 2021) <http://ise.thss.tsinghua.edu.cn/~mlong/doc/Representation-Subspace-Distance-for-Domain-Adaptation-Regression-icml21.pdf>`_

    Args:
        trade_off (float):  The trade-off value between Representation Subspace Distance
            and Base Mismatch Penalization. Default: 0.1

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    """
    def __init__(self, trade_off=0.1):
        super(RepresentationSubspaceDistance, self).__init__()
        self.trade_off = trade_off

    def forward(self, f_s, f_t):
        U_s, _, _ = torch.svd(f_s.t())
        U_t, _, _ = torch.svd(f_t.t())
        P_s, cosine, P_t = torch.svd(torch.mm(U_s.t(), U_t))
        sine = torch.sqrt(1 - torch.pow(cosine, 2))
        rsd = torch.norm(sine, 1)  # Representation Subspace Distance
        bmp = torch.norm(torch.abs(P_s) - torch.abs(P_t), 2)  # Base Mismatch Penalization
        return rsd + self.trade_off * bmp