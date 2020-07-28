from typing import Optional, Sequence, Tuple
import copy
import torch
from torch import Tensor


def accuracy(output, target, topk=(1,)) -> Sequence[float]:
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Inputs:
        - **output** (Tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        - **target** (Tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        - **topk** (sequence[int]): A list of top-N number.

    Outputs: res
        - **res** (sequence[float]): Top-N accuracies (N :math:`\in` K).
    """
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


def partial_accuracy(output: Tensor, target: Tensor, included: Optional[Sequence] = None, excluded: Optional[Sequence] = None) -> Tuple[int, int]:
    r"""
    Computes the accuracy over the top predictions for the specified classes.

    Inputs:
        - **output** (Tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        - **target** (Tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        - **included** (sequence[int], optional): The class index to include when calculating accuracy
        - **excluded** (sequence[int], optional): The class index to exclude when calculating accuracy

    Outputs: (correct, batch_size)
        - **correct** (int): Number of correct samples
        - **batch_size** (int): Number of valid samples (after including or excluding specific classes)

    .. note::
       When `included` is provided, `excluded` is ignored.

       When `included` and `excluded` are None, partial_accuracy is equal to `accuracy`.

    """
    target = copy.deepcopy(target)
    with torch.no_grad():
        if included is not None:
            for i, v in enumerate(target):
                if v not in included:
                    target[i] = -1
        elif excluded is not None:
            for i, v in enumerate(target):
                if v in excluded:
                    target[i] = -1
        batch_size = int(torch.sum(target != -1))

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct = correct[:1].view(-1).float().sum(0, keepdim=True)
        correct = correct.mul_(100.0 / batch_size) if batch_size != 0 else torch.zeros(1)
        return correct, batch_size

