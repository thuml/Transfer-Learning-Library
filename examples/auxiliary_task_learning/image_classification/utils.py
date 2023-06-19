"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import sys
import time

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.append('../..')
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.data import ForeverDataIterator
import tllib.vision.datasets as datasets
from tllib.weighting.task_sampler import *


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        train_source_datasets = {domain_name: dataset(root=root, task=domain_name, split='train', download=True,
                                                      transform=train_source_transform) for domain_name in source}
        train_target_datasets = {domain_name: dataset(root=root, task=domain_name, split='train', download=True,
                                                      transform=train_target_transform) for domain_name in target}
        val_datasets = {
            domain_name: dataset(root=root, task=domain_name, split='val', download=True, transform=val_transform) for
            domain_name in target}
        test_datasets = {
            domain_name: dataset(root=root, task=domain_name, split='test', download=True, transform=val_transform) for
            domain_name in target}
        class_names = list(val_datasets.values())[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    print("source train:", {domain_name: len(dataset) for domain_name, dataset in train_source_datasets.items()})
    print("target train:", {domain_name: len(dataset) for domain_name, dataset in train_target_datasets.items()})
    print("target val:", {domain_name: len(dataset) for domain_name, dataset in val_datasets.items()})
    print("target test:", {domain_name: len(dataset) for domain_name, dataset in test_datasets.items()})
    return train_source_datasets, train_target_datasets, val_datasets, test_datasets, num_classes, class_names


def validate(task_name, val_loader, model, args, device, verbose=True) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images, task_name)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and verbose:
                progress.display(i)

        if verbose:
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
            if confmat:
                print(confmat.format(args.class_names))

    return top1.avg


def validate_all(val_loaders, model, args, device, verbose=True):
    acc_dict = {}
    for task_name, data_loader in val_loaders.items():
        acc_dict[task_name] = validate(task_name, data_loader, model, args, device, verbose)
    return acc_dict


def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False,
                        resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def train(task_sampler, model, optimizer,
          epoch, iters_per_epoch, args, device):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    accs = AverageMeter('Acc', ':3.1f')
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, losses, accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(iters_per_epoch):
        task_name, dataloader = task_sampler.pop()
        x, labels = next(dataloader)[:2]

        x = x.to(device)
        labels = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y = model(x, task_name)

        loss = F.cross_entropy(y, labels)

        acc = accuracy(y, labels)[0]

        losses.update(loss.item(), x.size(0))
        accs.update(acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def get_task_sampler(datasets, args):
    data_loaders = {name: ForeverDataIterator(DataLoader(dataset, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.workers, drop_last=True)) for
                    name, dataset in datasets.items()}
    task_to_num_examples_dict = {
        task_name: len(dataset) for task_name, dataset in datasets.items()
    }
    if args.examples_cap is None:
        args.examples_cap = max(task_to_num_examples_dict.values())
    if args.sampler == "UniformMultiTaskSampler":
        task_sampler = UniformMultiTaskSampler(data_loaders, rng=0)
    elif args.sampler == "ProportionalMultiTaskSampler":
        task_sampler = ProportionalMultiTaskSampler(data_loaders, rng=0,
                                                    task_to_num_examples_dict=task_to_num_examples_dict)
        print("Task prob:", task_sampler.task_p)
    elif args.sampler == "TemperatureMultiTaskSampler":
        task_sampler = TemperatureMultiTaskSampler(
            data_loaders, rng=0, task_to_num_examples_dict=task_to_num_examples_dict,
            temperature=args.temperature, examples_cap=args.examples_cap
        )
        print("Task prob:", task_sampler.task_p)
    else:
        raise NotImplementedError(args.sampler)
    return task_sampler
