from torch import nn
from .layers import ConvBlock


__all__ = ['FCN']


class FCN(nn.Module):
    def __init__(self, in_features, channels=[128, 256, 128], kernel_sizes=[7, 5, 3]):
        super(FCN, self).__init__()
        assert len(channels) == len(kernel_sizes)
        self.convblock1 = ConvBlock(in_features, channels[0], kernel_sizes[0], 1)
        self.convblock2 = ConvBlock(channels[0], channels[1], kernel_sizes[1], 1)
        self.convblock3 = ConvBlock(channels[1], channels[2], kernel_sizes[2], 1)
        self.channels = channels

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

    @property
    def out_features(self):
        return self.channels[-1]


def fcn(in_features, **kwargs):
    return FCN(in_features, **kwargs)

