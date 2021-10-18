"""
Modified from https://github.com/Simon4Yan/eSPGAN
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic block with structure Conv-LeakyReLU->Pool"""
    def __init__(self, in_dim, out_dim):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_block(x)


class SiameseNetwork(nn.Module):
    """Siamese network whose input is an image of shape :math:`(3,H,W)` and output is an one-dimensional feature vector.

    Args:
        nsf (int): dimension of output feature representation.
    """
    def __init__(self, nsf=64):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, nsf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(nsf, nsf * 2),
            ConvBlock(nsf * 2, nsf * 4),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, nsf * 2, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(nsf * 2, nsf, bias=False)

    def forward(self, x):
        x = self.flatten(self.conv(x))
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x)
        return x
