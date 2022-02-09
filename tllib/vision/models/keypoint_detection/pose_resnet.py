"""
Modified from https://github.com/microsoft/human-pose-estimation.pytorch
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn
from ..resnet import _resnet, Bottleneck


class Upsampling(nn.Sequential):
    """
    3-layers deconvolution used in `Simple Baseline <https://arxiv.org/abs/1804.06208>`_.
    """
    def __init__(self, in_channel=2048, hidden_dims=(256, 256, 256), kernel_sizes=(4, 4, 4), bias=False):
        assert len(hidden_dims) == len(kernel_sizes), \
            'ERROR: len(hidden_dims) is different len(kernel_sizes)'

        layers = []
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise NotImplementedError("kernel_size is {}".format(kernel_size))

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_channel = hidden_dim

        super(Upsampling, self).__init__(*layers)

        # init following Simple Baseline
        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PoseResNet(nn.Module):
    """
    `Simple Baseline <https://arxiv.org/abs/1804.06208>`_ for keypoint detection.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints, finetune=False):
        super(PoseResNet, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        self.finetune = finetune
        for m in self.head.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.upsampling(x)
        x = self.head(x)
        return x

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
        ]


def _pose_resnet(arch, num_keypoints, block, layers, pretrained_backbone, deconv_with_bias, finetune=False, progress=True, **kwargs):
    backbone = _resnet(arch, block, layers, pretrained_backbone, progress, **kwargs)
    upsampling = Upsampling(backbone.out_features, bias=deconv_with_bias)
    model = PoseResNet(backbone, upsampling, 256, num_keypoints, finetune)
    return model


def pose_resnet101(num_keypoints, pretrained_backbone=True, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    """Constructs a Simple Baseline model with a ResNet-101 backbone.

    Args:
        num_keypoints (int): number of keypoints
        pretrained_backbone (bool, optional): If True, returns a model pre-trained on ImageNet. Default: True.
        deconv_with_bias (bool, optional): Whether use bias in the deconvolution layer. Default: False
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True
    """
    return _pose_resnet('resnet101', num_keypoints, Bottleneck, [3, 4, 23, 3], pretrained_backbone, deconv_with_bias, finetune, progress, **kwargs)