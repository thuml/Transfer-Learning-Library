import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

__all__ = ['StochNorm1d', 'StochNorm2d']


class _StochNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_StochNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.p = 0.0
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        if self.training:
            z_0 = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                False, self.momentum, self.eps)

            z_1 = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                True, self.momentum, self.eps)

            if input.dim() == 2:
                s = torch.from_numpy(
                    np.random.binomial(n=1, p=self.p, size=self.num_features).reshape(1,
                                                                                      self.num_features)).float().cuda()
            elif input.dim() == 3:
                s = torch.from_numpy(
                    np.random.binomial(n=1, p=self.p, size=self.num_features).reshape(1, self.num_features,
                                                                                      1)).float().cuda()
            elif input.dim() == 4:
                s = torch.from_numpy(
                    np.random.binomial(n=1, p=self.p, size=self.num_features).reshape(1, self.num_features, 1,
                                                                                      1)).float().cuda()
            else:
                raise BaseException()

            z = (1 - s) * z_0 + s * z_1
        else:
            z = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                False, self.momentum, self.eps)

        return z



class StochNorm1d(_StochNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class StochNorm2d(_StochNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class StochNorm3d(_StochNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))



def convert_model(module, p):
    """Traverse the input module and its child recursively
       and replace all instance of torch.nn.modules.batchnorm.BatchNorm*N*d
       to StochNorm*N*d
    Args:
        module: the input module needs to be convert to StochNorm model
    """
    # if isinstance(module, torch.nn.DataParallel):
    #     mod = module.module
    #     mod = convert_model(mod)
    #     mod = DataParallelWithCallback(mod, device_ids=module.device_ids)
    #     return mod

    mod = module
    for pth_module, stoch_module in zip([torch.nn.modules.batchnorm.BatchNorm1d,
                                        torch.nn.modules.batchnorm.BatchNorm2d,
                                        torch.nn.modules.batchnorm.BatchNorm3d],
                                       [StochNorm1d,
                                        StochNorm2d,
                                        StochNorm3d]):
        if isinstance(module, pth_module):
            mod = stoch_module(module.num_features, module.eps, module.momentum, module.affine)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            mod.p = p
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child, p))

    return mod