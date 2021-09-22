"""
Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import torch.nn as nn
import functools
from .util import get_norm_layer, init_weights


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Args:
            dim (int): the number of channels in the conv layer.
            padding_type (str): the name of padding layer: reflect | replicate | zero
            norm_layer (torch.nn.Module): normalization layer
            use_dropout (bool): if use dropout layers.
            use_bias (bool): if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)

    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Args:
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            ngf (int): the number of filters in the last conv layer
            norm_layer (torch.nn.Module): normalization layer
            use_dropout (bool): if use dropout layers
            n_blocks (int): the number of ResNet blocks
            padding_type (str): the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Args:
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            num_downs (int): the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int): the number of filters in the last conv layer
            norm_layer(torch.nn.Module): normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Args:
            outer_nc (int): the number of filters in the outer conv layer
            inner_nc (int): the number of filters in the inner conv layer
            input_nc (int): the number of channels in input images/features
            submodule (UnetSkipConnectionBlock): previously defined submodules
            outermost (bool): if this module is the outermost module
            innermost (bool): if this module is the innermost module
            norm_layer (torch.nn.Module): normalization layer
            use_dropout (bool): if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


def resnet_9(ngf, input_nc=3, output_nc=3, norm='batch', use_dropout=False,
                       init_type='normal', init_gain=0.02):
    """
    Resnet-based generator with 9 Resnet blocks.

    Args:
        ngf (int): the number of filters in the last conv layer
        input_nc (int): the number of channels in input images. Default: 3
        output_nc (int): the number of channels in output images. Default: 3
        norm (str): the type of normalization layers used in the network. Default: 'batch'
        use_dropout (bool): whether use dropout. Default: False
        init_type (str): the name of the initialization method. Choices includes: ``normal`` |
            ``xavier`` | ``kaiming`` | ``orthogonal``. Default: 'normal'
        init_gain (float): scaling factor for normal, xavier and orthogonal. Default: 0.02
    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    init_weights(net, init_type, init_gain)
    return net


def resnet_6(ngf, input_nc=3, output_nc=3, norm='batch', use_dropout=False,
                       init_type='normal', init_gain=0.02):
    """
    Resnet-based generator with 6 Resnet blocks.

    Args:
        ngf (int): the number of filters in the last conv layer
        input_nc (int): the number of channels in input images. Default: 3
        output_nc (int): the number of channels in output images. Default: 3
        norm (str): the type of normalization layers used in the network. Default: 'batch'
        use_dropout (bool): whether use dropout. Default: False
        init_type (str): the name of the initialization method. Choices includes: ``normal`` |
            ``xavier`` | ``kaiming`` | ``orthogonal``. Default: 'normal'
        init_gain (float): scaling factor for normal, xavier and orthogonal. Default: 0.02
    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    init_weights(net, init_type, init_gain)
    return net


def unet_256(ngf, input_nc=3, output_nc=3, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02):
    """
    `U-Net <https://arxiv.org/abs/1505.04597>`_ generator for 256x256 input images.
    The size of the input image should be a multiple of 256.

    Args:
        ngf (int): the number of filters in the last conv layer
        input_nc (int): the number of channels in input images. Default: 3
        output_nc (int): the number of channels in output images. Default: 3
        norm (str): the type of normalization layers used in the network. Default: 'batch'
        use_dropout (bool): whether use dropout. Default: False
        init_type (str): the name of the initialization method. Choices includes: ``normal`` |
            ``xavier`` | ``kaiming`` | ``orthogonal``. Default: 'normal'
        init_gain (float): scaling factor for normal, xavier and orthogonal. Default: 0.02

    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    init_weights(net, init_type, init_gain)
    return net


def unet_128(ngf, input_nc=3, output_nc=3, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02):
    """
    `U-Net <https://arxiv.org/abs/1505.04597>`_ generator for 128x128 input images.
    The size of the input image should be a multiple of 128.

    Args:
        ngf (int): the number of filters in the last conv layer
        input_nc (int): the number of channels in input images. Default: 3
        output_nc (int): the number of channels in output images. Default: 3
        norm (str): the type of normalization layers used in the network. Default: 'batch'
        use_dropout (bool): whether use dropout. Default: False
        init_type (str): the name of the initialization method. Choices includes: ``normal`` |
            ``xavier`` | ``kaiming`` | ``orthogonal``. Default: 'normal'
        init_gain (float): scaling factor for normal, xavier and orthogonal. Default: 0.02

    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    init_weights(net, init_type, init_gain)
    return net


def unet_32(ngf, input_nc=3, output_nc=3, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02):
    """
    `U-Net <https://arxiv.org/abs/1505.04597>`_ generator for 32x32 input images

    Args:
        ngf (int): the number of filters in the last conv layer
        input_nc (int): the number of channels in input images. Default: 3
        output_nc (int): the number of channels in output images. Default: 3
        norm (str): the type of normalization layers used in the network. Default: 'batch'
        use_dropout (bool): whether use dropout. Default: False
        init_type (str): the name of the initialization method. Choices includes: ``normal`` |
            ``xavier`` | ``kaiming`` | ``orthogonal``. Default: 'normal'
        init_gain (float): scaling factor for normal, xavier and orthogonal. Default: 0.02

    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    init_weights(net, init_type, init_gain)
    return net