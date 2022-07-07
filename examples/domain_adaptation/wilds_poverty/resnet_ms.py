"""
Modified based on torchvision.models.resnet
@author: Jiaxin Li
@contact: thulijx@gmail.com
"""
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck
import copy

__all__ = ['resnet18_ms', 'resnet34_ms', 'resnet50_ms', 'resnet101_ms', 'resnet152_ms']


class ResNetMS(models.ResNet):
    """
    ResNet with input channels parameter, without fully connected layer.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super(ResNetMS, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self._out_features = self.fc.in_features
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)


def resnet18_ms(num_channels=3):
    model = ResNetMS(num_channels, BasicBlock, [2, 2, 2, 2])
    return model


def resnet34_ms(num_channels=3):
    model = ResNetMS(num_channels, BasicBlock, [3, 4, 6, 3])
    return model


def resnet50_ms(num_channels=3):
    model = ResNetMS(num_channels, Bottleneck, [3, 4, 6, 3])
    return model


def resnet101_ms(num_channels=3):
    model = ResNetMS(num_channels, Bottleneck, [3, 4, 23, 3])
    return model


def resnet152_ms(num_channels=3):
    model = ResNetMS(num_channels, Bottleneck, [3, 8, 36, 3])
    return model
