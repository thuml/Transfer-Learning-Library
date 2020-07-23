import sys
from .metric import partial_accuracy


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
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
    def __init__(self, classes, fmt=':f'):
        self.fmt = fmt
        self.classes = classes
        self.acc_meters = [AverageMeter(class_name, fmt) for class_name in classes]

    def reset(self):
        for acc_meter in self.acc_meters:
            acc_meter.reset()

    def update(self, output, target):
        for i, acc_meter in enumerate(self.acc_meters):
            correct, batch_size = partial_accuracy(output, target, included=[i])
            acc_meter.update(correct.item(), batch_size)

    def accuracy(self, class_name: str):
        idx = self.classes.index(class_name)
        if self.acc_meters[idx].count > 0:
            return self.acc_meters[idx].avg
        else:
            return None

    def average_accuracy(self, class_names):
        sum = 0
        count = 0
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
        return fmtstr.format(name="accuracy", val=self.average_accuracy(self.classes))
