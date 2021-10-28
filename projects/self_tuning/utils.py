"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import sys
import time
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

sys.path.append('../..')
from ssllib.rand_augment import RandAugment
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter

sys.path.append('.')
import datasets


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrained_checkpoint=None):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=True)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    if pretrained_checkpoint:
        print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
        pretrained_dict = torch.load(pretrained_checkpoint)
        backbone.load_state_dict(pretrained_dict, strict=False)
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )


def get_dataset(dataset_name, root, sample_rate, train_transform, val_transform, unlabeled_train_transform=None,
                num_samples=None):
    if unlabeled_train_transform is None:
        unlabeled_train_transform = train_transform
    if dataset_name == 'CIFAR100':
        assert num_samples is not None
        base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=True)
        labeled_idxes, unlabeled_idxes = x_u_split(num_samples, 100, base_dataset.targets)
        labeled_train_dataset = datasets.CIFAR100(root, labeled_idxes, train=True,
                                                  transform=train_transform)
        unlabeled_train_dataset = datasets.CIFAR100(root, unlabeled_idxes, train=True,
                                                    transform=unlabeled_train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root, train=False, transform=val_transform, download=False)
    else:
        dataset = datasets.__dict__[dataset_name]
        labeled_train_dataset = dataset(root=root, split='train', sample_rate=sample_rate,
                                        download=True, transform=train_transform)
        unlabeled_train_dataset = dataset(root=root, split='train', sample_rate=sample_rate,
                                          download=True, transform=unlabeled_train_transform, unlabeled=True)
        val_dataset = dataset(root=root, split='test', sample_rate=100, download=True, transform=val_transform)
    return labeled_train_dataset, unlabeled_train_dataset, val_dataset


def x_u_split(num_samples, num_classes, labels):
    assert num_samples % num_classes == 0
    n_samples_per_class = int(num_samples / num_classes)
    labels = np.array(labels)

    labeled_idxes = []
    for i in range(num_classes):
        ith_class_idxes = np.where(labels == i)[0]
        ith_class_idxes = np.random.choice(ith_class_idxes, n_samples_per_class, False)
        labeled_idxes.extend(ith_class_idxes)

    unlabeled_idxes = [i for i in range(len(labels)) if i not in labeled_idxes]
    return labeled_idxes, unlabeled_idxes


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

    return top1.avg


def get_train_transform(resizing='default', random_horizontal_flip=True, rand_augment=False,
                        norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
        ])
    elif resizing == 'cifar':
        transform = T.Compose([
            T.RandomCrop(size=32, padding=4, padding_mode='reflect'),
            ResizeImage(224)
        ])
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if rand_augment:
        transforms.append(RandAugment(n=2, m=10))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'cifar':
        transform = ResizeImage(224)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
