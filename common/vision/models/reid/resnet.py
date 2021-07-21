from common.vision.models.resnet import ResNet, load_state_dict_from_url, model_urls, BasicBlock, Bottleneck

__all__ = ['reid_resnet50']


class ReidResNet(ResNet):

    def __init__(self, *args, **kwargs):
        super(ReidResNet, self).__init__(*args, **kwargs)
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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


def reid_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _reid_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                        **kwargs)
