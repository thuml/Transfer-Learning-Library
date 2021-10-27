"""
Modified from https://github.com/SikaStar/IDM
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from .idm import IDM
import torch.nn as nn
from collections import OrderedDict
from common.vision.models.reid.resnet import ReidResNet as ResNetBase
from common.vision.models.reid.resnet import (
    load_state_dict_from_url, model_urls, BasicBlock, Bottleneck
)

__all__ = ['reid_resnet18', 'reid_resnet34', 'reid_resnet50', 'reid_resnet101']


class ReidResNet(ResNetBase):
    r"""Modified `ResNet` architecture with `IDM` module from `IDM: An Intermediate Domain Module for Domain Adaptive
    Person Re-ID (ICCV 2021) <https://arxiv.org/pdf/2108.02413v1.pdf>`_. Although `IDM` Module can be inserted anywhere,
    original paper places `IDM` after layer0-4. Our implementation follows this idea, but you are free to modify this
    function to try other possibilities.
    """
    def __init__(self, *args, **kwargs):
        super(ReidResNet, self).__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            OrderedDict([
                ('conv1', self.conv1),
                ('bn1', self.bn1),
                ('relu', self.relu),
                ('maxpool', self.maxpool)
            ])
        )

        self.idm_layer1 = IDM(channel=64)
        self.idm_layer2 = IDM(channel=256)
        self.idm_layer3 = IDM(channel=512)
        self.idm_layer4 = IDM(channel=1024)
        self.idm_layer5 = IDM(channel=2048)

    def forward(self, x, stage=0):
        assert stage in range(5)

        x = self.conv(x)
        if stage == 0 and self.training:
            x, attention_lam = self.idm_layer1(x)
        x = self.layer1(x)
        if stage == 1 and self.training:
            x, attention_lam = self.idm_layer2(x)
        x = self.layer2(x)
        if stage == 2 and self.training:
            x, attention_lam = self.idm_layer3(x)
        x = self.layer3(x)
        if stage == 3 and self.training:
            x, attention_lam = self.idm_layer4(x)
        x = self.layer4(x)
        if stage == 4 and self.training:
            x, attention_lam = self.idm_layer5(x)

        if self.training:
            return x, attention_lam
        else:
            return x


def _reid_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ReidResNet(block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                                   progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def reid_resnet18(pretrained=False, progress=True, **kwargs):
    r"""Constructs a Reid-ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _reid_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                        **kwargs)


def reid_resnet34(pretrained=False, progress=True, **kwargs):
    r"""Constructs a Reid-ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _reid_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                        **kwargs)


def reid_resnet50(pretrained=False, progress=True, **kwargs):
    r"""Constructs a Reid-ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _reid_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                        **kwargs)


def reid_resnet101(pretrained=False, progress=True, **kwargs):
    r"""Constructs a Reid-ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _reid_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                        **kwargs)
