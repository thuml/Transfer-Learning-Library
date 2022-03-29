"""
@author: Jinghan Gao
@contact: getterk@163.com
"""
import argparse
import sys
import time
from typing import Dict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import *

from tllib.alignment.dann import ImageClassifier, DomainAdversarialLoss
from tllib.utils import ForeverDataIterator, AverageMeter, ProgressMeter
from tllib.utils.metric import accuracy
from tllib.vision.datasets.universal import default_universal as universal

sys.path.append('../../..')
import tllib.vision.datasets.openset as datasets


class AccuracyCounter:

    def __init__(self, length):
        self.Ncorrect = np.zeros(length)
        self.Ntotal = np.zeros(length)
        self.length = length

    def add_correct(self, index, amount=1):
        self.Ncorrect[index] += amount

    def add_total(self, index, amount=1):
        self.Ntotal[index] += amount

    def clear_zero(self):
        i = np.where(self.Ntotal == 0)
        self.Ncorrect = np.delete(self.Ncorrect, i)
        self.Ntotal = np.delete(self.Ntotal, i)

    def each_accuracy(self):
        self.clear_zero()
        return self.Ncorrect / self.Ntotal

    def mean_accuracy(self):
        self.clear_zero()
        return np.mean(self.Ncorrect / self.Ntotal)

    def h_score(self):
        self.clear_zero()
        common_acc = np.mean(self.Ncorrect[0:-1] / self.Ntotal[0:-1])
        open_acc = self.Ncorrect[-1] / self.Ntotal[-1]
        return 2 * common_acc * open_acc / (common_acc + open_acc)


class Ensemble(nn.Module):

    def __init__(self, in_feature, num_classes):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(in_feature, num_classes)
        self.fc2 = nn.Linear(in_feature, num_classes)
        self.fc3 = nn.Linear(in_feature, num_classes)
        self.fc4 = nn.Linear(in_feature, num_classes)
        self.fc5 = nn.Linear(in_feature, num_classes)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc5.weight)

    def forward(self, x, index):
        if index == 0:
            y = self.fc1(x)
            # y = nn.Softmax(dim=-1)(y_1)
        elif index == 1:
            y = self.fc2(x)
            # y = nn.Softmax(dim=-1)(y_2)
        elif index == 2:
            y = self.fc3(x)
            # y = nn.Softmax(dim=-1)(y_3)
        elif index == 3:
            y = self.fc4(x)
            # y = nn.Softmax(dim=-1)(y_4)
        elif index == 4:
            y = self.fc5(x)
            # y = nn.Softmax(dim=-1)(y_5)
        else:
            y_1 = self.fc1(x)
            y_1 = nn.Softmax(dim=-1)(y_1)
            y_2 = self.fc2(x)
            y_2 = nn.Softmax(dim=-1)(y_2)
            y_3 = self.fc3(x)
            y_3 = nn.Softmax(dim=-1)(y_3)
            y_4 = self.fc4(x)
            y_4 = nn.Softmax(dim=-1)(y_4)
            y_5 = self.fc5(x)
            y_5 = nn.Softmax(dim=-1)(y_5)
            return y_1, y_2, y_3, y_4, y_5

        return y

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.parameters(), "lr_mult": 1.},
        ]
        return params


esem_transforms = [Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, interpolation=InterpolationMode.BICUBIC,
                 fill=(255, 255, 255)),
    CenterCrop(224),
    RandomGrayscale(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
]), Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[0]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
]), Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, interpolation=InterpolationMode.BICUBIC,
                 fill=(255, 255, 255)),
    FiveCrop(224),
    Lambda(lambda crops: crops[1]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
]), Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, interpolation=InterpolationMode.BICUBIC,
                 fill=(255, 255, 255)),
    RandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[2]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
]), Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[3]),
    RandomGrayscale(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])]


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    # load datasets from tllib.vision.datasets
    dataset = datasets.__dict__[dataset_name]
    source_dataset = universal(dataset, source=True)
    target_dataset = universal(dataset, source=False)

    train_source_dataset = source_dataset(root=root, task=source, download=True, transform=train_source_transform)
    train_target_dataset = target_dataset(root=root, task=target, download=True, transform=train_target_transform)
    esem_datasets = [source_dataset(root=root, task=source, download=True, transform=esem_transforms[i]) for i in
                     range(5)]
    val_dataset = target_dataset(root=root, task=target, download=True, transform=val_transform)
    if dataset_name == 'DomainNet':
        test_dataset = target_dataset(root=root, task=target, split='test', download=True, transform=val_transform)
    else:
        test_dataset = val_dataset
    class_names = np.unique(train_source_dataset.targets)
    num_classes = len(class_names)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, esem_datasets, num_classes, class_names


def get_esem_data_iters(esem_datasets, batch_size, num_workers):
    return [ForeverDataIterator(
        DataLoader(esem_datasets[i], batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True))
        for i in range(5)]


def get_marginal_confidence(y_1, y_2, y_3, y_4, y_5):
    conf_1, _ = torch.topk(y_1, 2, 1)
    conf_2, _ = torch.topk(y_2, 2, 1)
    conf_3, _ = torch.topk(y_3, 2, 1)
    conf_4, _ = torch.topk(y_4, 2, 1)
    conf_5, _ = torch.topk(y_5, 2, 1)

    conf_1 = conf_1[:, 0] - conf_1[:, 1]
    conf_2 = conf_2[:, 0] - conf_2[:, 1]
    conf_3 = conf_3[:, 0] - conf_3[:, 1]
    conf_4 = conf_4[:, 0] - conf_4[:, 1]
    conf_5 = conf_5[:, 0] - conf_5[:, 1]

    confidence = (conf_1 + conf_2 + conf_3 + conf_4 + conf_5) / 5
    return confidence


def get_entropy(y_1, y_2, y_3, y_4, y_5):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy2 = torch.sum(- y_2 * torch.log(y_2 + 1e-10), dim=1)
    entropy3 = torch.sum(- y_3 * torch.log(y_3 + 1e-10), dim=1)
    entropy4 = torch.sum(- y_4 * torch.log(y_4 + 1e-10), dim=1)
    entropy5 = torch.sum(- y_5 * torch.log(y_5 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))

    entropy = (entropy1 + entropy2 + entropy3 + entropy4 + entropy5) / (5 * entropy_norm)
    return entropy


def norm(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x


def cal_pr(scores, labels, thresholds):
    new_scores = zip(scores.numpy(), labels)
    new_scores = np.array(sorted(new_scores, key=lambda x: x[0], reverse=True))

    points = []
    for threshold in thresholds:
        TP, FP, FN = 0, 0, 0
        for score, label in new_scores:
            if score >= threshold:
                if label:
                    TP += 1
                else:
                    FP += 1
            else:
                if label:
                    FN += 1
        points.append([TP / (TP + FN + 1e-7), TP / (TP + FP + 1e-7)])

    print_points = [[0., 1.]] + sorted(points, key=lambda x: x[0]) + [[1., 0.]]
    return list(zip(*print_points)), points


def cal_f1(val_loader, model, esem, source_classes, device):
    # switch to evaluate mode
    model.eval()
    esem.eval()

    # all_confidence = list()
    all_marginal_confidence = list()
    all_entropy = list()
    all_labels = list()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)

            _, f = model(images)
            yt_1, yt_2, yt_3, yt_4, yt_5 = esem(f)
            marginal_confidence = get_marginal_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)

            # all_confidence.extend(confidence)
            all_marginal_confidence.extend(marginal_confidence)
            all_entropy.extend(entropy)
            all_labels.extend(labels)

    all_marginal_confidence = norm(torch.tensor(all_marginal_confidence))
    all_entropy = norm(torch.tensor(all_entropy))
    all_scores = (all_marginal_confidence + 1 - all_entropy) / 2

    common_labels = [(1 if label in source_classes else 0) for label in all_labels]
    common_labels = np.array(common_labels)

    step = 0.05
    thresholds = [thresh * step for thresh in range(int(1 / step) - 1, -1, -1)]

    f1s = []
    _, points = cal_pr(all_scores, common_labels, thresholds)
    for j, point in enumerate(points):
        recall, precision = point
        f1s.append((2 * recall * precision) / (recall + precision + 1e-7))

    return max(f1s)


def evaluate_source_common(val_loader, model, esem, source_classes, args, device):
    temperature = 1
    # switch to evaluate mode
    model.eval()
    esem.eval()

    common = []
    target_private = []

    all_confidence = list()
    all_entropy = list()
    all_labels = list()
    all_output = list()

    source_weight = torch.zeros(len(source_classes)).to(device)
    cnt = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)

            output, f = model(images)
            output = F.softmax(output, -1) / temperature
            yt_1, yt_2, yt_3, yt_4, yt_5 = esem(f)
            confidence = get_marginal_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)

            all_confidence.extend(confidence)
            all_entropy.extend(entropy)
            all_labels.extend(labels)

            for each_output in output:
                all_output.append(each_output)

    all_confidence = norm(torch.tensor(all_confidence))
    all_entropy = norm(torch.tensor(all_entropy))
    all_score = (all_confidence + 1 - all_entropy) / 2

    print('source_threshold = {}'.format(args.src_threshold))

    for i in range(len(all_score)):
        if all_score[i] >= args.src_threshold:
            source_weight += all_output[i]
            cnt += 1
        if all_labels[i] in source_classes:
            common.append(all_score[i])
        else:
            target_private.append(all_score[i])

    hist, bin_edges = np.histogram(common, bins=20, range=(0, 1))
    print(hist)

    hist, bin_edges = np.histogram(target_private, bins=20, range=(0, 1))
    print(hist)

    source_weight = norm(source_weight / cnt)
    print('---source_weight---')
    print(source_weight)
    return source_weight


def pretrain(train_source_iter: ForeverDataIterator, esem_iters, model, esem, optimizer, args, epoch, lr_scheduler,
             device):
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, cls_accs],
        prefix="Pre: [{}]".format(epoch))

    model.train()
    esem.train()

    for i in range(args.iters_per_epoch):
        lr_scheduler.step()

        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)
        y_s, f_s = model(x_s)
        cls_loss = F.cross_entropy(y_s, labels_s)

        esem_losses = []
        for i, esem_iter in enumerate(esem_iters):
            x_s1, labels_s1 = next(esem_iter)
            x_s1 = x_s1.to(device)
            labels_s1 = labels_s1.to(device)
            y_s1, f_s1 = model(x_s1)
            y_s1 = esem(f_s1, index=i)
            esem_losses.append(F.cross_entropy(y_s1, labels_s1))

        cls_acc = accuracy(y_s1, labels_s1)[0]
        cls_accs.update(cls_acc.item(), args.batch_size)

        loss = sum(esem_losses) + cls_loss
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (args.print_freq) == 0:
            progress.display(i)


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, esem, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, source_class_weight, target_score_upper, target_score_lower,
          args: argparse.Namespace, device):
    batch_time = AverageMeter('Time', ':4.2f')
    losses = AverageMeter('Loss', ':4.2f')
    cls_accs = AverageMeter('Cls Acc', ':4.1f')
    domain_accs = AverageMeter('Domain Acc', ':4.1f')
    score_upper = AverageMeter('Score Upper', ':4.2f')
    score_lower = AverageMeter('Score Lower', ':4.2f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, losses, cls_accs, domain_accs, score_upper, score_lower],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()
    esem.eval()

    end = time.time()
    for i in range(args.iters_per_epoch):
        lr_scheduler.step()

        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        with torch.no_grad():
            yt_1, yt_2, yt_3, yt_4, yt_5 = esem(f_t)
            confidence = get_marginal_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)
            w_t = (confidence + 1 - entropy) / 2
            target_score_upper = target_score_upper * 0.01 + w_t.max() * 0.99
            target_score_lower = target_score_lower * 0.01 + w_t.min() * 0.99
            w_t = (w_t - target_score_lower) / (target_score_upper - target_score_lower)
            w_s = torch.tensor([source_class_weight[i] for i in labels_s]).to(device)

        # f_s_normed = F.normalize(f_s, dim=1)
        # f_s_normed_1, f_s_normed_2 = torch.split(f_s_normed, [args.batch_size, args.batch_size], dim=0)
        # f_s_normed = torch.cat([f_s_normed_1.unsqueeze(1), f_s_normed_2.unsqueeze(1)], dim=1)
        loss = F.cross_entropy(y_s, labels_s)  # + scloss(f_s_normed, labels_s)
        transfer_loss = domain_adv(f_s, f_t, w_s.detach(), w_t.detach())
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss += transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), args.batch_size)
        cls_accs.update(cls_acc.item(), args.batch_size)
        domain_accs.update(domain_acc.item(), args.batch_size)
        score_upper.update(target_score_upper.item(), 1)
        score_lower.update(target_score_lower.item(), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return target_score_upper, target_score_lower


def train_esem(train_source_iter, model, esem, optimizer, lr_scheduler, epoch, args, index, device):
    losses = AverageMeter('Loss', ':4.2f')
    cls_accs = AverageMeter('Cls Acc', ':5.1f')
    progress = ProgressMeter(
        args.iters_per_epoch // 2,
        [losses, cls_accs],
        prefix="Esem: [{}-{}]".format(epoch, index))

    model.eval()
    esem.train()

    for i in range(args.iters_per_epoch // 2):
        lr_scheduler.step()

        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # compute output
        with torch.no_grad():
            y_s, f_s = model(x_s)
        y_s = esem(f_s.detach(), index)

        loss = F.cross_entropy(y_s, labels_s)
        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, esem, source_classes, args, device):
    # switch to evaluate mode
    model.eval()
    esem.eval()

    all_confidence = list()
    all_entropy = list()
    all_indices = list()
    all_labels = list()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            output, f = model(images)
            values, indices = torch.max(F.softmax(output, -1), 1)

            yt_1, yt_2, yt_3, yt_4, yt_5 = esem(f)
            confidence = get_marginal_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)

            all_confidence.extend(confidence)
            all_entropy.extend(entropy)
            all_indices.extend(indices)
            all_labels.extend(labels)

    all_confidence = norm(torch.tensor(all_confidence))
    all_entropy = norm(torch.tensor(all_entropy))
    all_score = (all_confidence + 1 - all_entropy) / 2

    counters = AccuracyCounter(len(source_classes) + 1)
    for (each_indice, each_label, score) in zip(all_indices, all_labels, all_score):
        if each_label in source_classes:
            counters.add_total(each_label)
            if score >= args.threshold and each_indice == each_label:
                counters.add_correct(each_label)
        else:
            counters.add_total(-1)
            if score < args.threshold:
                counters.add_correct(-1)

    print('---counters---')
    print(counters.each_accuracy())
    print(counters.mean_accuracy())
    print(counters.h_score())

    return counters.mean_accuracy(), counters.h_score()
