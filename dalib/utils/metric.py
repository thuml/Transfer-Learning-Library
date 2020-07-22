from typing import Optional, Sequence
import copy
import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = int(torch.sum(target != -1))

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def partial_accuracy(output, target, included: Optional[Sequence] = None, exclueded: Optional[Sequence] = None):
    """Computes the accuracy over the top predictions for the specified classes"""
    target = copy.deepcopy(target)
    with torch.no_grad():
        if included is not None:
            for i, v in enumerate(target):
                if v not in included:
                    target[i] = -1
        elif exclueded is not None:
            for i, v in enumerate(target):
                if v in exclueded:
                    target[i] = -1
        batch_size = int(torch.sum(target != -1))

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct = correct[:1].view(-1).float().sum(0, keepdim=True)
        correct = correct.mul_(100.0 / batch_size) if batch_size != 0 else torch.zeros(1)
        return correct, batch_size

