"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class AdaAggLayer(nn.Module):
    r"""Applies an adaptive aggregate conv2d to the incoming data:.`
    """
    __constants__ = ['in_planes', 'out_planes', 'kernel_size', 'experts']

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, experts=5, align=True, lite=False):
        super(AdaAggLayer, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.experts = experts
        self.align = align
        self.lite = lite
        self.m = 0.1

        self.weight = nn.Parameter(torch.randn(experts, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(experts, out_planes))
        else:
            self.bias = None

        # channel-wise align
        if self.align and self.kernel_size > 1:
            align_conv = torch.zeros(self.experts * out_planes, out_planes, 1, 1)

            for i in range(self.experts):
                for j in range(self.out_planes):
                    align_conv[i * self.out_planes + j, j, 0, 0] = 1.0

            self.align_conv = nn.Parameter(align_conv, requires_grad=True)
        else:
            self.align = False

        # lite version
        if self.lite:
            self.register_buffer('lite_attention', torch.zeros(self.experts))

        # attention layer
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // 4 + 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 4 + 1, experts, 1, bias=True),
            nn.Flatten(),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.attention:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for i in range(self.experts):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        sigmoid_attention = self.attention(x)  # batch_size * experts

        batch_size = x.shape[0]

        # lite version
        if self.lite:
            if self.training:
                sigmoid_attention = sigmoid_attention.mean(0)
                self.lite_attention = (1 - self.m) * self.lite_attention + self.m * sigmoid_attention
            else:
                sigmoid_attention = self.lite_attention

            sigmoid_attention = sigmoid_attention.unsqueeze(0).repeat(batch_size, 1)

        # x = x.view(1, -1, height, width)   # 1 * BC * H * W
        x = rearrange(x, '(d b) c h w->d (b c) h w', d=1)

        # channel-wise align
        if self.align:
            weight = rearrange(self.weight, '(d e) o i j k->d (e o) i (j k)', d=1)
            # weight = self.weight.view(1, self.experts * self.out_planes, self.in_planes, self.kernel_size * self.kernel_size)
            weight = F.conv2d(weight, weight=self.align_conv, bias=None, stride=1, padding=0, dilation=1, groups=self.experts)
            weight = rearrange(weight, 'd (e o) i (j k)->(d e) o i j k', e=self.experts, j=self.kernel_size)
        else:
            weight = self.weight

        # weight = self.weight
        aggregate_weight = rearrange(
            torch.einsum('be,eoijk->boijk', sigmoid_attention, weight),
            'b o i j k->(b o) i j k'
        )

        if self.bias is not None:
            aggregate_bias = torch.einsum('be,eo->bo', sigmoid_attention, self.bias).view(-1)
            y = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            y = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        # y = y.view(batch_size, self.out_planes, y.size(-2), y.size(-1))
        y = rearrange(y, 'd (b o) h w->(d b) o h w', d=1, b=batch_size)
        
        return y


experts = 5

__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']


def conv7x7(in_planes, out_planes, padding=3, stride=1, groups=1, dilation=1, experts=5, align=False, lite=False):
    return AdaAggLayer(
        in_planes, out_planes, kernel_size=7, stride=stride, 
        padding=padding, groups=groups, bias=False, dilation=dilation,
        experts=experts, align=align, lite=lite
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, experts=5, align=False, lite=False):
    return AdaAggLayer(
        in_planes, out_planes, kernel_size=3, stride=stride, 
        padding=dilation, groups=groups, bias=False, dilation=dilation,
        experts=experts, align=align, lite=lite
    )


def conv1x1(in_planes, out_planes, stride=1, experts=5, align=False, lite=False):
    return AdaAggLayer(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False, 
        experts=experts, align=align, lite=lite
    )

# basicblock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, experts=5, align=False, lite=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, experts=experts, align=align, lite=lite)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, experts=experts, align=align, lite=lite)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, experts=experts, align=align, lite=lite)
        self.bn3 = norm_layer(planes * self.expansion) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, experts=5, align=True, lite=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.experts = experts
        self.align = align
        self.lite = lite

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv7x7(3, self.inplanes, stride=2, padding=3, experts=experts, align=align, lite=lite)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 
                                       experts=experts, lite=lite)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], 
                                       experts=experts, lite=lite)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], 
                                       experts=experts, align=align, lite=lite)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], 
                                       experts=experts, align=align, lite=lite)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._out_features = 512 * block.expansion
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, experts=5, align=False, lite=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            experts=experts, align=align, lite=lite))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                experts=experts, align=align, lite=lite))

        return nn.Sequential(*layers)
    
    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features
    
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
        # x = x.reshape(x.size(0), -1)
        # x = self.fc(x)

        return x

