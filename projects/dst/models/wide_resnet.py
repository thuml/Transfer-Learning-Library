"""
Modified from https://github.com/xternalz/WideResNet-pytorch,
https://github.com/TorchSSL/TorchSSL/blob/main/models/nets/wrn_var.py
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.equal_in_out) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                   padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equal_in_out and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equal_in_out else self.conv_shortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """WideResNet with 3 residual blocks."""

    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)

        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self._out_features = channels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        return out

    @property
    def out_features(self):
        return self._out_features


class WideResNetVar(nn.Module):
    """A variant of WideResNet with 4 residual blocks."""

    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNetVar, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor, 128 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 2, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # 4th block
        self.block4 = NetworkBlock(
            n, channels[3], channels[4], block, 2, drop_rate)

        self.bn1 = nn.BatchNorm2d(channels[4], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self._out_features = channels[4]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.relu(self.bn1(out))

        return out

    @property
    def out_features(self):
        return self._out_features
