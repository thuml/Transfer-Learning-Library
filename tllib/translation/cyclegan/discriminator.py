"""
Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn
from torch.nn import init
import functools
from .util import get_norm_layer, init_weights


class NLayerDiscriminator(nn.Module):
    """Construct a PatchGAN discriminator

    Args:
        input_nc (int): the number of channels in input images.
        ndf (int): the number of filters in the last conv layer. Default: 64
        n_layers (int): the number of conv layers in the discriminator. Default: 3
        norm_layer (torch.nn.Module): normalization layer. Default: :class:`nn.BatchNorm2d`
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Construct a 1x1 PatchGAN discriminator (pixelGAN)

    Args:
        input_nc (int): the number of channels in input images.
        ndf (int): the number of filters in the last conv layer. Default: 64
        norm_layer (torch.nn.Module): normalization layer. Default: :class:`nn.BatchNorm2d`
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


def patch(ndf, input_nc=3, norm='batch', n_layers=3, init_type='normal', init_gain=0.02):
    """
    PatchGAN classifier described in the original pix2pix paper.
    It can classify whether 70×70 overlapping patches are real or fake.
    Such a patch-level discriminator architecture has fewer parameters
    than a full-image discriminator and can work on arbitrarily-sized images
    in a fully convolutional fashion.

    Args:
        ndf (int): the number of filters in the first conv layer
        input_nc (int): the number of channels in input images. Default: 3
        norm (str): the type of normalization layers used in the network. Default: 'batch'
        n_layers (int): the number of conv layers in the discriminator. Default: 3
        init_type (str): the name of the initialization method. Choices includes: ``normal`` |
            ``xavier`` | ``kaiming`` | ``orthogonal``. Default: 'normal'
        init_gain (float): scaling factor for normal, xavier and orthogonal. Default: 0.02
    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers, norm_layer=norm_layer)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def pixel(ndf, input_nc=3, norm='batch', init_type='normal', init_gain=0.02):
    """
    1x1 PixelGAN discriminator can classify whether a pixel is real or not.
    It encourages greater color diversity but has no effect on spatial statistics.

    Args:
        ndf (int): the number of filters in the first conv layer
        input_nc (int): the number of channels in input images. Default: 3
        norm (str): the type of normalization layers used in the network. Default: 'batch'
        init_type (str): the name of the initialization method. Choices includes: ``normal`` |
            ``xavier`` | ``kaiming`` | ``orthogonal``. Default: 'normal'
        init_gain (float): scaling factor for normal, xavier and orthogonal. Default: 0.02
    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    init_weights(net, init_type, init_gain=init_gain)
    return net