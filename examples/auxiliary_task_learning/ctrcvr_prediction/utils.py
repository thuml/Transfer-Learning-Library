import tqdm
import time

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from tllib.utils.meter import AverageMeter, ProgressMeter

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, epoch, device, args, task_weights=None):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    loss_meters = {task_name: AverageMeter("Loss({})".format(task_name), ":5.2f")
                   for task_name in args.train_tasks}
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time] + list(loss_meters.values()),
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    criterion = torch.nn.BCELoss()

    end = time.time()
    for i, (features, labels) in enumerate(data_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        x = [features["categorical"].to(device), features["numerical"].to(device)]
        labels = {k: v.to(device) for k, v in labels.items()}
        y = model(x)

        losses = {}
        for task_name in args.train_tasks:
            losses[task_name] = criterion(y[task_name].squeeze(-1), labels[task_name].float())
            loss_meters[task_name].update(losses[task_name])

        if task_weights is None:
            loss = sum(losses.values())
        else:
            loss = sum([losses[task_name] * task_weights[task_name] for task_name in args.train_tasks])

        model.zero_grad()
        loss.backward()
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def test(model, data_loader, device, args):
    batch_time = AverageMeter('Time', ':5.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time],
        prefix="Testing: "
    )

    model.eval()
    all_labels = {task_name: [] for task_name in args.test_tasks}
    all_predictions = {task_name: [] for task_name in args.test_tasks}
    end = time.time()
    with torch.no_grad():
        for i, (features, labels) in enumerate(data_loader):
            x = [features["categorical"].to(device), features["numerical"].to(device)]
            labels = {k: v.to(device) for k, v in labels.items()}
            y = model(x)
            for task_name in args.test_tasks:
                all_labels[task_name].extend(labels[task_name].tolist())
                all_predictions[task_name].extend(y[task_name].squeeze(-1).tolist())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return {task_name: roc_auc_score(all_labels[task_name], all_predictions[task_name]) for task_name in args.test_tasks}
