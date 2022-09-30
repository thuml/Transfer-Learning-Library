"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import time
import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from rand_augment import RandAugment
import datasets
import models

sys.path.append('../..')
from tllib.modules.classifier import Classifier as ClassifierBase
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.metric import accuracy, ConfusionMatrix

# dataset statistics
CIFAR100_MEAN = (0.507, 0.487, 0.441)
CIFAR100_STD = (0.267, 0.256, 0.276)
CIFAR10_MEAN = (0.491, 0.482, 0.447)
CIFAR10_STD = (0.247, 0.244, 0.262)
SVHN_MEAN = (0.438, 0.444, 0.473)
SVHN_STD = (0.175, 0.177, 0.174)
STL10_MEAN = (0.441, 0.428, 0.387)
STL10_STD = (0.268, 0.261, 0.269)


def get_train_transform(img_size, random_horizontal_flip=True, rand_augment=False, norm_mean=CIFAR10_MEAN,
                        norm_std=CIFAR10_STD):
    transforms = [T.Resize(img_size)]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if rand_augment:
        transforms.append(RandAugment(n=3, m=5))
    transforms.extend([
        T.RandomCrop(img_size, padding=4, padding_mode='reflect'),
        T.ToTensor(),
        T.Normalize(norm_mean, norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(img_size, norm_mean=CIFAR10_MEAN, norm_std=CIFAR10_STD):
    transforms = [
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(norm_mean, norm_std)
    ]
    return T.Compose(transforms)


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )


def sample_labeled_data(num_samples_per_class, num_classes, labels, seed):
    """Randomly sample labeled subset. The selected subset is **deterministic** given the same random seed.

    Args:
        num_samples_per_class (int): number of labeled samples for each class
        num_classes (int): number of classes
        labels (list): label information
        seed (int): random seed

    """

    labels = np.array(labels)
    assert num_samples_per_class * num_classes <= len(labels)
    random_state = np.random.RandomState(seed)

    labeled_idxes = []
    for i in range(num_classes):
        ith_class_idxes = np.where(labels == i)[0]
        ith_class_idxes = random_state.choice(ith_class_idxes, num_samples_per_class, False)
        labeled_idxes.extend(ith_class_idxes)

    return np.array(labeled_idxes)


def get_dataset(dataset_name, root, num_samples_per_class, batch_size, labeled_train_transform, val_transform,
                unlabeled_train_transform=None, seed=0):
    def duplicate_labeled_indexes(idxes):
        """
        When mini-batch size is larger than the size of labeled subset in SSL, we simply duplicate selected idxes.
        """

        if len(idxes) < batch_size:
            n_expand = math.ceil(batch_size / len(idxes))
            idxes = np.hstack([idxes for _ in range(n_expand)])
        return idxes

    if unlabeled_train_transform is None:
        unlabeled_train_transform = labeled_train_transform

    # We adopt the split schema of FixMatch, which is detailed as follows
    if dataset_name == 'STL10':
        """
        split for STL10
            
            labeled data: sampled from 'train' split
            unlabeled data: all data from 'train' split and 'unlabeled' split
        """
        dataset = datasets.__dict__[dataset_name]
        base_dataset = dataset(root=root, split='train', transform=labeled_train_transform, download=True)
        num_classes = base_dataset.num_classes

        # randomly sample labeled data
        labeled_idxes = sample_labeled_data(num_samples_per_class, num_classes, base_dataset.targets, seed=seed)
        labeled_idxes = duplicate_labeled_indexes(labeled_idxes)
        # labeled subset
        labeled_train_dataset = Subset(base_dataset, labeled_idxes)
        # unlabeled_subset
        stl10_train_split = dataset(root=root, split='train', transform=unlabeled_train_transform, download=True)
        stl10_unlabeled_split = dataset(root=root, split='unlabeled', transform=unlabeled_train_transform,
                                        download=True)
        unlabeled_train_dataset = ConcatDataset([stl10_train_split, stl10_unlabeled_split])
        # val dataset
        val_dataset = dataset(root=root, split='test', transform=val_transform, download=True)
    elif dataset_name == 'SVHN':
        """
        split for SVHN
            
            labeled data: sampled from 'train' split and 'extra' split
            unlabeled data: all data from 'train' split and 'extra' split
        """
        dataset = datasets.__dict__[dataset_name]
        svhn_train_split = dataset(root=root, split='train', transform=labeled_train_transform, download=True)
        num_classes = svhn_train_split.num_classes
        svhn_extra_split = dataset(root=root, split='extra', transform=labeled_train_transform, download=True)

        base_dataset = ConcatDataset([svhn_train_split, svhn_extra_split])
        targets = np.concatenate([svhn_train_split.targets, svhn_extra_split.targets])

        # randomly sample labeled data
        labeled_idxes = sample_labeled_data(num_samples_per_class, num_classes, targets, seed=seed)
        labeled_idxes = duplicate_labeled_indexes(labeled_idxes)
        # labeled subset
        labeled_train_dataset = Subset(base_dataset, labeled_idxes)
        # unlabeled subset
        unlabeled_train_dataset = ConcatDataset([
            dataset(root=root, split='train', transform=unlabeled_train_transform, download=True),
            dataset(root=root, split='extra', transform=unlabeled_train_transform, download=True)
        ])
        # val dataset
        val_dataset = dataset(root=root, split='test', transform=val_transform, download=True)
    else:
        """
        split for CIFAR10 and CIFAR100
            
            labeled data: sampled from 'train' split
            unlabeled data: all data from 'train' split
        """
        dataset = datasets.__dict__[dataset_name]
        base_dataset = dataset(root=root, split='train', transform=labeled_train_transform, download=True)
        num_classes = base_dataset.num_classes

        # randomly sample labeled data
        labeled_idxes = sample_labeled_data(num_samples_per_class, base_dataset.num_classes,
                                            base_dataset.targets, seed=seed)
        labeled_idxes = duplicate_labeled_indexes(labeled_idxes)
        # labeled subset
        labeled_train_dataset = Subset(base_dataset, labeled_idxes)
        # unlabeled subset
        unlabeled_train_dataset = dataset(root=root, split='train', transform=unlabeled_train_transform, download=True)
        # val dataset
        val_dataset = dataset(root=root, split='test', transform=val_transform, download=True)

    return labeled_train_dataset, unlabeled_train_dataset, val_dataset, num_classes


def get_model_names():
    return sorted(name for name in models.__dict__)


def get_model(model_name, **kwargs):
    backbone = models.__dict__[model_name](**kwargs)
    return backbone


class Classifier(ClassifierBase):

    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        y = self.head(f)
        return y

    def get_parameters(self, lr, weight_decay):
        if self.finetune:
            raise NotImplementedError

        # no weight decay on bias and BN parameters
        wd_params = []
        no_wd_params = []
        for name, param in self.named_parameters():
            if 'bias' in name or 'bn' in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        params = [
            {"params": wd_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_wd_params, "lr": lr, "weight_decay": 0.}
        ]

        return params


def unwrap_ddp(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def update_bn(model, ema_model):
    """Update BN statistics of the EMA model."""
    model_unwrap = unwrap_ddp(model)
    eam_model_unwrap = ema_model.teacher if hasattr(ema_model, 'teacher') else ema_model

    for m2, m1 in zip(eam_model_unwrap.named_modules(), model_unwrap.named_modules()):
        if ('bn' in m2[0]) and ('bn' in m1[0]):
            bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
            bn2['running_mean'].data.copy_(bn1['running_mean'].data)
            bn2['running_var'].data.copy_(bn1['running_var'].data)
            bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def validate(val_loader, ema_model, args):
    batch_time = AverageMeter('time', ':3.3f')
    cls_accs = AverageMeter('acc@1', ':3.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, cls_accs],
        prefix='test: ')

    # switch to eval mode
    ema_model.eval()
    confmat = ConfusionMatrix(args.num_classes)

    print('evaluating')
    with torch.no_grad():
        end = time.time()
        for i, (x, labels) in enumerate(val_loader):
            x = x.cuda()
            labels = labels.cuda()

            # compute output
            y = ema_model(x)
            if args.distributed:
                y = concat_all_gather(y)
                labels = concat_all_gather(labels)

            cls_acc, = accuracy(y, labels, topk=(1,))
            cls_accs.update(cls_acc.item(), x.size(0))
            confmat.update(labels, y.argmax(1))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    if args.local_rank == 0:
        _, per_class_acc, _ = confmat.compute()
        np.save(os.path.join(args.log, 'per_class_acc'), per_class_acc.cpu().numpy())

    print('acc@1 {cls_accs.avg:.3f}'.format(cls_accs=cls_accs))
    return cls_accs.avg


class DistributedProxySampler(DistributedSampler):
    """Distributed sampler which does not implement shuffling.
    Adopted from https://github.com/TorchSSL/TorchSSL/blob/main/datasets/DistributedProxySampler.py.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with log_softmax, which is more numerically stable."""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y, labels, reduction='mean'):
        log_p = torch.log_softmax(y, dim=1)
        return F.nll_loss(log_p, labels, reduction=reduction)


class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """Self-training loss with confidence threshold."""

    def __init__(self, threshold: float):
        super(ConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold
        self.criterion = CrossEntropyLoss()

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = (self.criterion(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels
