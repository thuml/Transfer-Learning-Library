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


class ClassifierRegularization(nn.Module):
    def __init__(self, parameters: list):
        super(ClassifierRegularization, self).__init__()
        self.parameters = parameters
    def forward(self):
        output = 0.0
        for name, param in self.parameters:
            output += 0.5 * torch.norm(param) ** 2
        return output


class L2spRegularization(nn.Module):
    def __init__(self, backbone_source: nn.Module, backbone_target: nn.Module):
        super(L2spRegularization, self).__init__()
        self.backbone_target = backbone_target
        self.source_weight = {}
        for name, param in backbone_source.named_parameters():
            self.source_weight[name] = param.detach()

    def forward(self):
        output = 0.0
        for name, param in self.backbone_target.named_parameters():
            output += 0.5 * torch.norm(param - self.source_weight[name]) ** 2
        return output


class FeatureRegularization(nn.Module):
    def __init__(self):
        super(FeatureRegularization, self).__init__()

    def forward(self, layer_outputs_source, layer_outputs_target):
        output = 0.0
        for fm_src, fm_tgt in zip(layer_outputs_source, layer_outputs_target):
            fm_src = layer_outputs_source[fm_src]
            fm_tgt = layer_outputs_target[fm_tgt]
            output += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
        return output


class AttentionFeatureRegularization(nn.Module):
    def __init__(self, channel_weight):
        super(AttentionFeatureRegularization, self).__init__()
        self.channel_weight = channel_weight

    def forward(self, layer_outputs_source, layer_outputs_target):
        output = 0.0
        for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source, layer_outputs_target)):
            fm_src = layer_outputs_source[fm_src]
            fm_tgt = layer_outputs_target[fm_tgt]

            b, c, h, w = fm_src.shape
            fm_src = fm_src.reshape(b, c, h * w)
            fm_tgt = fm_tgt.reshape(b, c, h * w)

            distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
            distance = c * torch.mul(self.channel_weight[i], distance ** 2) / (h * w)
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


class ChannelWeightCalculator:
    def __init__(self, model, return_layers, criterion, data_loader, device, iteration_limit=-1):
        self._model = model
        self.return_layers = return_layers
        self.channel_weight = []
        self.criterion = criterion
        self.data_loader = data_loader
        self.iteration_limit = iteration_limit
        self.device = device
        for layer_id, name in enumerate(self.return_layers):
            layer = get_attribute(self._model, name)
            layer_channel_weight = [0] * layer.out_channels
            self.channel_weight.append(layer_channel_weight)

    def train_classifier(self, optimizer, scheduler, num_epochs):
        iterations_per_epoch = len(self.data_loader)
        self._model.train()
        self._model.backbone.requires_grad = False

        for epoch in range(num_epochs):
            losses = AverageMeter('Loss', ':3.2f')
            cls_accs = AverageMeter('Cls Acc', ':3.1f')
            progress = ProgressMeter(
                iterations_per_epoch,
                [losses, cls_accs],
                prefix="Epoch: [{}]".format(epoch))

            for i, data in enumerate(self.data_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self._model(inputs)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cls_acc = accuracy(outputs, labels)[0]

                losses.update(loss.item(), inputs.size(0))
                cls_accs.update(cls_acc.item(), inputs.size(0))

                if i % 10 == 0:
                    progress.display(i)
            scheduler.step()

    def calculate(self):
        print('Calculating channel weights...')
        self._model.eval()

        if self.iteration_limit > 0:
            total_iteration = min(len(self.data_loader), self.iteration_limit)
        else:
            total_iteration = len(self.data_loader)

        progress = ProgressMeter(
            total_iteration,
            [],
            prefix="Iteration: ")

        for i, data in enumerate(self.data_loader):
            if i >= total_iteration:
                break
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs, _ = self._model(inputs)
            loss_0 = self.criterion(outputs, labels)
            progress.display(i)
            for layer_id, name in enumerate(tqdm(self.return_layers)):
                layer = get_attribute(self._model, name)
                for j in range(layer.out_channels):
                    tmp = self._model.state_dict()[name + '.weight'][j, ].clone()
                    self._model.state_dict()[name + '.weight'][j, ] = 0.0
                    outputs, _ = self._model(inputs)
                    loss_1 = self.criterion(outputs, labels)
                    difference = loss_1 - loss_0
                    difference = difference.detach().cpu().numpy().item()
                    history_value = self.channel_weight[layer_id][j]
                    self.channel_weight[layer_id][j] = 1.0 * (i * history_value + difference) / (i + 1)
                    self._model.state_dict()[name + '.weight'][j, ] = tmp

    def save_channel_weight(self, file_path):
        json.dump(self.channel_weight, open(file_path, 'w'))


def calculate_channel_weight(classifier, criterion, optimizer, loader, scheduler, return_layers, device, channel_weight_path, args):
    calculator = ChannelWeightCalculator(classifier, return_layers=return_layers,
                                         criterion=criterion, data_loader=loader, device=device, iteration_limit=args.iteration_limit)
    calculator.train_classifier(optimizer=optimizer, scheduler=scheduler,
                                num_epochs=args.epochs_channel_weight)
    calculator.calculate()
    calculator.save_channel_weight(channel_weight_path)


def get_channel_weight(file_path, device):
    channel_weights = []
    for weight in json.load(open(file_path)):
        weight = np.array(weight)
        weight = (weight - np.mean(weight)) / np.std(weight)
        weight = torch.from_numpy(weight).float().to(device)
        weight = F.softmax(weight / 5).detach()
        channel_weights.append(weight)
    return channel_weights
