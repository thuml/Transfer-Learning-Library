"""
Modified from https://github.com/SikaStar/IDM
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn


class IDM(nn.Module):
    """IDM module in `IDM: An Intermediate Domain Module for Domain Adaptive Person Re-ID (ICCV 2021)
    <https://arxiv.org/pdf/2108.02413v1.pdf>`_. This module automatically generates mixed features
    with source and target features.
    """
    def __init__(self, channel=64):
        super(IDM, self).__init__()
        self.channel = channel
        self.fc1 = nn.Linear(channel * 2, channel)
        self.fc2 = nn.Linear(channel, int(channel / 2))
        self.fc3 = nn.Linear(int(channel / 2), 2)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        if not self.training:
            return x

        bs = x.size(0)
        assert bs % 2 == 0
        split = torch.split(x, int(bs / 2), 0)
        x_s = split[0].contiguous()
        x_t = split[1].contiguous()

        x_emb_s = torch.cat((self.avgpool(x_s.detach()).squeeze(), self.maxpool(x_s.detach()).squeeze()), 1)
        x_emb_t = torch.cat((self.avgpool(x_t.detach()).squeeze(), self.maxpool(x_t.detach()).squeeze()), 1)

        x_emb_s, x_emb_t = self.fc1(x_emb_s), self.fc1(x_emb_t)
        x_emb = x_emb_s + x_emb_t
        x_emb = self.fc2(x_emb)
        lam = self.fc3(x_emb)
        lam = self.softmax(lam)
        x_inter = lam[:, 0].reshape(-1, 1, 1, 1) * x_s + lam[:, 1].reshape(-1, 1, 1, 1) * x_t
        output = torch.cat((x_s, x_t, x_inter), 0)

        return output, lam
