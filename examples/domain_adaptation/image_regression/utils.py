"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import sys
import time
import torch
import torch.nn.functional as F

sys.path.append('../../..')
from common.utils.meter import AverageMeter, ProgressMeter
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.modules.instancenorm import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d


def convert_model(module):
    """convert BatchNorms in the `module` into InstanceNorms"""
    source_modules = (BatchNorm1d, BatchNorm2d, BatchNorm3d)
    target_modules = (InstanceNorm1d, InstanceNorm2d, InstanceNorm3d)
    for src_module, tgt_module in zip(source_modules, target_modules):
        if isinstance(module, src_module):
            mod = tgt_module(module.num_features, module.eps, module.momentum, module.affine)
            module = mod

    for name, child in module.named_children():
        module.add_module(name, convert_model(child))

    return module


def validate(val_loader, model, args, factors, device):
    batch_time = AverageMeter('Time', ':6.3f')
    mae_losses = [AverageMeter('mae {}'.format(factor), ':6.3f') for factor in factors]
    progress = ProgressMeter(
        len(val_loader),
        [batch_time] + mae_losses,
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            for j in range(len(factors)):
                mae_loss = F.l1_loss(output[:, j], target[:, j])
                mae_losses[j].update(mae_loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        for i, factor in enumerate(factors):
            print("{} MAE {mae.avg:6.3f}".format(factor, mae=mae_losses[i]))
        mean_mae = sum(l.avg for l in mae_losses) / len(factors)
    return mean_mae

