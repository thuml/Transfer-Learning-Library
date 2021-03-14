import torch
import torch.nn as nn
import functools
from collections import OrderedDict


def flatten_outputs(fea):
    return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))


def reg_att_fea_map(inputs, layer_outputs_source, layer_outputs_target, model_source, channel_weights):
    fea_loss = 0.0
    for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source, layer_outputs_target)):
        fm_src = layer_outputs_source[fm_src]
        fm_tgt = layer_outputs_source[fm_tgt]
        b, c, h, w = fm_src.shape
        fm_src = flatten_outputs(fm_src)
        fm_tgt = flatten_outputs(fm_tgt)
        div_norm = h * w
        distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
        distance = c * torch.mul(channel_weights[i], distance ** 2) / (h * w)
        fea_loss += 0.5 * torch.sum(distance)
    return fea_loss


class ClassifierRegLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super(ClassifierRegLoss, self).__init__()
        self.model = model

    def forward(self):
        output = 0.0
        for name, param in self.model.head.named_parameters():
            output += 0.5 * torch.norm(param) ** 2
        for name, param in self.model.bottleneck.named_parameters():
            output += 0.5 * torch.norm(param) ** 2
        return output


class L2spRegLoss(nn.Module):
    def __init__(self, source_model: nn.Module, target_model: nn.Module):
        super(L2spRegLoss, self).__init__()
        self.source_model = source_model
        self.target_model = target_model
        self.source_weight = {}
        for name, param in self.source_model.backbone.named_parameters():
            self.source_weight[name] = param.detach()

    def forward(self):
        output = 0.0
        for name, param in self.target_model.backbone.named_parameters():
            output += 0.5 * torch.norm(param - self.source_weight[name]) ** 2
        return output


class FeatureRegLoss(nn.Module):
    def __init__(self):
        super(FeatureRegLoss, self).__init__()

    def forward(self, layer_outputs_source, layer_outputs_target):
        output = 0.0
        for fm_src, fm_tgt in zip(layer_outputs_source, layer_outputs_target):
            fm_src = layer_outputs_source[fm_src]
            fm_tgt = layer_outputs_target[fm_tgt]
            output += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
        return output


def rgetattr(obj, attr, *args):
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
            layer = rgetattr(self._model, name)
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
