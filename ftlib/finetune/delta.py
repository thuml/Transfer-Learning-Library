import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from collections import OrderedDict

from tqdm import tqdm

from common.utils import AverageMeter, ProgressMeter
from common.utils.metric import accuracy


class L2Regularization(nn.Module):
    def __init__(self, model: nn.Module):
        super(L2Regularization, self).__init__()
        self.model = model

    def forward(self):
        output = 0.0
        for param in self.model.parameters():
            output += 0.5 * torch.norm(param) ** 2
        return output


class L2SPRegularization(nn.Module):
    def __init__(self, source_model: nn.Module, target_model: nn.Module):
        super(L2SPRegularization, self).__init__()
        self.target_model = target_model
        self.source_weight = {}
        for name, param in source_model.named_parameters():
            self.source_weight[name] = param.detach()

    def forward(self):
        output = 0.0
        for name, param in self.target_model.named_parameters():
            output += 0.5 * torch.norm(param - self.source_weight[name]) ** 2
        return output


class FeatureRegularization(nn.Module):
    def __init__(self):
        super(FeatureRegularization, self).__init__()

    def forward(self, layer_outputs_source, layer_outputs_target):
        output = 0.0
        for fm_src, fm_tgt in zip(layer_outputs_source.values(), layer_outputs_target.values()):
            output += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
        return output


class AttentionFeatureRegularization(nn.Module):
    def __init__(self, channel_attention):
        super(AttentionFeatureRegularization, self).__init__()
        self.channel_attention = channel_attention

    def forward(self, layer_outputs_source, layer_outputs_target):
        output = 0.0
        for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source.values(), layer_outputs_target.values())):
            b, c, h, w = fm_src.shape
            fm_src = fm_src.reshape(b, c, h * w)
            fm_tgt = fm_tgt.reshape(b, c, h * w)

            distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
            distance = c * torch.mul(self.channel_attention[i], distance ** 2) / (h * w)
            output += 0.5 * torch.sum(distance)

        return output


def get_attribute(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class IntermediateLayerGetter:
    def __init__(self, model, return_layers, keep_output=True):
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output

    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name in self.return_layers:
            layer = get_attribute(self._model, name)
            def hook(module, input, output, name=name):
                ret[name] = output
            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')
            handles.append(h)

        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None

        for h in handles:
            h.remove()

        return ret, output
