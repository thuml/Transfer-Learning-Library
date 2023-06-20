"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append('../..')
from tllib.utils.meter import AverageMeter, ProgressMeter

class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, pred, gt):
        return self.loss_fn(pred, gt.long())


class DepthEstimationLoss(nn.Module):
    def __init__(self):
        super(DepthEstimationLoss, self).__init__()

    def forward(self, pred, gt):
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = torch.sum(torch.abs(pred - gt) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return loss


class SurfaceNormalPredictionLoss(nn.Module):
    def __init__(self):
        super(SurfaceNormalPredictionLoss, self).__init__()

    def forward(self, pred, gt):
        # gt has been normalized on the NYUv2 dataset
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = 1 - torch.sum((pred * gt) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return loss


class SegmentationMetric:
    def __init__(self, num_classes=13):
        super(SegmentationMetric, self).__init__()

        self.num_classes = num_classes
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def update(self, pred, gt):
        self.record = self.record.to(pred.device)
        pred = pred.softmax(1).argmax(1).flatten()
        gt = gt.long().flatten()
        k = (gt >= 0) & (gt < self.num_classes)
        inds = self.num_classes * gt[k].to(torch.int64) + pred[k]
        self.record += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def compute(self):
        h = self.record.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        acc = torch.diag(h).sum() / h.sum()
        return [torch.mean(iu).item(), acc.item()]

    def reset(self):
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def __str__(self):
        iou, acc = self.compute()
        return "Seg mIoU: {:.2%} Pix Acc: {:.2%}".format(iou, acc)

# depth
class DepthEstimationMetric:
    def __init__(self):
        super(DepthEstimationMetric, self).__init__()
        self.abs_record = []
        self.rel_record = []
        self.bs = []

    def update(self, pred, gt):
        device = pred.device
        binary_mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1).to(device)
        pred = pred.masked_select(binary_mask)
        gt = gt.masked_select(binary_mask)
        abs_err = torch.abs(pred - gt)
        rel_err = torch.abs(pred - gt) / gt
        abs_err = (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
        rel_err = (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
        self.abs_record.append(abs_err)
        self.rel_record.append(rel_err)
        self.bs.append(pred.size()[0])

    def compute(self):
        records = np.stack([np.array(self.abs_record), np.array(self.rel_record)])
        batch_size = np.array(self.bs)
        return [(records[i] * batch_size).sum() / (sum(batch_size)) for i in range(2)]

    def reset(self):
        self.abs_record = []
        self.rel_record = []
        self.bs = []

    def __str__(self):
        abs_err, rel_err = self.compute()
        return "Depth Abs Err: {:5.2f} Rel Err: {:5.2f}".format(abs_err, rel_err)

# normal
class SurfaceNormalPredictionMetric:
    def __init__(self):
        super(SurfaceNormalPredictionMetric, self).__init__()
        self.record = []
        self.bs = []
    def update(self, pred, gt):
        # gt has been normalized on the NYUv2 dataset
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        binary_mask = (torch.sum(gt, dim=1) != 0)
        error = torch.acos(
            torch.clamp(torch.sum(pred * gt, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        self.record.append(error)

    def compute(self):
        records = np.concatenate(self.record)
        return [np.mean(records), np.median(records), \
                np.mean((records < 11.25) * 1.0), np.mean((records < 22.5) * 1.0), \
                np.mean((records < 30) * 1.0)]

    def reset(self):
        self.record = []
        self.bs = []

    def __str__(self):
        results = self.compute()
        return "Normal Mean: {:5.2f} Median: {:5.2f} <11.25: {:.2%} <22.5: {:.2%} <30: {:.2%}".format(*results)


def train(data_loader, model, optimizer, epoch, args, device, task_weights=None):
    loss_functions = {
        'segmentation': SegmentationLoss(),
        'depth': DepthEstimationLoss(),
        'normal': SurfaceNormalPredictionLoss(),
    }
    metric_functions = {
        'segmentation': SegmentationMetric(),
        'depth': DepthEstimationMetric(),
        'normal': SurfaceNormalPredictionMetric(),
    }

    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    loss_meters = {task_name: AverageMeter("Loss({})".format(task_name), ":5.2f")
                   for task_name in args.train_tasks}
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time] + list(loss_meters.values()) +
        [metric_functions[task_name] for task_name in args.train_tasks],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, labels) in enumerate(data_loader):
        x = x.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(x)

        losses = {}
        for task_name in args.train_tasks:
            output = outputs[task_name]
            prediction = F.interpolate(output, args.img_size, mode='bilinear', align_corners=True)
            losses[task_name] = loss_functions[task_name](prediction, labels[task_name])
            loss_meters[task_name].update(losses[task_name])
            metric_functions[task_name].update(prediction, labels[task_name])
        if task_weights is None:
            loss = sum(losses.values())
        else:
            loss = sum([losses[task_name] * task_weights[task_name] for task_name in args.train_tasks])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, args, device):
    loss_functions = {
        'segmentation': SegmentationLoss(),
        'depth': DepthEstimationLoss(),
        'normal': SurfaceNormalPredictionLoss(),
    }
    metric_functions = {
        'segmentation': SegmentationMetric(),
        'depth': DepthEstimationMetric(),
        'normal': SurfaceNormalPredictionMetric(),
    }
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    loss_meters = {task_name: AverageMeter("Loss({})".format(task_name), ":5.2f")
                   for task_name in args.test_tasks}
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time] + list(loss_meters.values()) +
        [metric_functions[task_name] for task_name in args.test_tasks],
        prefix="Test ")

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (x, labels) in enumerate(val_loader):
            x = x.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            outputs = model(x)

            losses = {}
            for task_name in args.test_tasks:
                output = outputs[task_name]
                prediction = F.interpolate(output, args.img_size, mode='bilinear', align_corners=True)
                losses[task_name] = loss_functions[task_name](prediction, labels[task_name])
                loss_meters[task_name].update(losses[task_name])
                metric_functions[task_name].update(prediction, labels[task_name])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return {task_name: metric_functions[task_name].compute() for task_name in args.test_tasks}


def count_improvement(base_result, new_result, indicator):
    """
    Calculates the improvement between two sets of results based on a specified indicator.

    Args:
        base_result: A dictionary containing the base result values for each task.
        new_result: A dictionary containing the new result values for each task.
        indicator: An indicator specifying whether a higher value indicates better performance for the k-th task.
                   Set to 0 if a higher value indicates better performance, and 1 otherwise.

    Returns:
        The calculated improvement value normalized by the number of tasks.
    """
    improvement = 0
    for task_name in new_result.keys():
        improvement += (((-1) ** np.array(indicator[task_name])) *
                        (- np.array(base_result[task_name]) + np.array(new_result[task_name])) /
                        np.array(base_result[task_name])).mean()
    return improvement / len(base_result)