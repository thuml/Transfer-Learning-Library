from typing import Optional, Sequence
from torch import Tensor
from .metric import partial_accuracy


class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ClassWiseAccuracyMeter:
    r"""Computes and stores the average and current accuracy for each class respectively.

    Parameters:
        - **classes** (sequence[str]): Names of all classes in the test dataset
        - **fmt** (string, optional): The format

    Examples::
        >>> # names of classes
        >>> classes = ["dog", "cat", "unknown"]
        >>> # Initialize a meter
        >>> accurcay_meter = ClassWiseAccuracyMeter(classes)
        >>> # Update meter after every minibatch update
        >>> accurcay_meter.update(output, target)
        >>> avg_accuracy = accurcay_meter.average_accuracy()
        >>> dog_accuracy = accurcay_meter.accuracy('dog')
        >>> unknown_accuracy = accurcay_meter.accuracy('unknown')
        >>> known_accuracy = accurcay_meter.average_accuracy(['dog', 'cat'])
    """
    def __init__(self, classes: Sequence[str], fmt: Optional[str] = ':f'):
        self.fmt = fmt
        self.classes = classes
        self.acc_meters = [AverageMeter(class_name, fmt) for class_name in classes]

    def reset(self):
        """Reset all accuracy meters to zero values"""
        for acc_meter in self.acc_meters:
            acc_meter.reset()

    def update(self, output: Tensor, target: Tensor):
        r"""
        First calculate accuracy for each class, then update the corresponding accuracy meter.

        Inputs:
            - **output** (Tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
            - **target** (Tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        """
        for i, acc_meter in enumerate(self.acc_meters):
            correct, batch_size = partial_accuracy(output, target, included=[i])
            acc_meter.update(correct.item(), batch_size)

    def accuracy(self, class_name: str):
        r"""
        Get the accuracy of a specific class.
        If there is no data for this class, then return None.
        """
        idx = self.classes.index(class_name)
        if self.acc_meters[idx].count > 0:
            return self.acc_meters[idx].avg
        else:
            return None

    def average_accuracy(self, class_names: Optional[Sequence[str]] = None):
        r"""
        Get the mean accuracy for partial classes specified by `class_names`.
        If there is no data for all classes in `class_names`, then return None.
        If `class_names` is None, then return the mean accuracy of all classes.
        """
        sum = 0
        count = 0
        if class_names is None:
            class_names = self.classes
        for class_name in class_names:
            accuracy = self.accuracy(class_name)
            if accuracy is not None:
                sum += accuracy
                count += 1
        if count > 0:
            return sum / count
        else:
            return None

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(name="accuracy", val=self.average_accuracy())
