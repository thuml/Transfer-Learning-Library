"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.modules.classifier import Classifier
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.metric import ConfusionMatrix, accuracy


class SequenceClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck=None,
                 bottleneck_dim=-1, head=None, finetune=False, pool_layer=None):
        if pool_layer is None:
            pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool1d(output_size=(1,)),
                nn.Flatten()
            )
        if bottleneck is None:
            bottleneck_dim = backbone.out_features
            bottleneck = nn.Sequential(
                nn.Linear(bottleneck_dim, bottleneck_dim),
                nn.ReLU(),
            )
        super(SequenceClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, head, finetune, pool_layer)


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))

    return top1.avg
