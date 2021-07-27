from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.modules.instancenorm import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d


def convert_model(module):
    source_modules = (BatchNorm1d, BatchNorm2d, BatchNorm3d)
    target_modules = (InstanceNorm1d, InstanceNorm2d, InstanceNorm3d)
    for src_module, tgt_module in zip(source_modules, target_modules):
        if isinstance(module, src_module):
            mod = tgt_module(module.num_features, module.eps, module.momentum, module.affine)
            module = mod

    for name, child in module.named_children():
        module.add_module(name, convert_model(child))

    return module