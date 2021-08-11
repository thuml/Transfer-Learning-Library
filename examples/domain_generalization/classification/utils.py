from typing import Tuple
import sys
import time
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import Subset, ConcatDataset

sys.path.append('../../..')
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter

supported_dataset = ['PACS', 'OfficeHome', 'DomainNet']


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name):
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
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )


def get_dataset(dataset_name, root, task_list, split='train', download=True, transform=None, seed=0) \
        -> Tuple[ConcatDataset, int]:
    assert dataset_name in supported_dataset
    assert split in ['train', 'val', 'all']
    dataset = datasets.__dict__[dataset_name]

    train_split_list = []
    val_split_list = []
    split_ratio = 0.8
    num_classes = 0

    for task in task_list:
        if dataset_name == 'PACS':
            domain = dataset(root=root, task=task, split='all', download=download, transform=transform)
            num_classes = domain.num_classes
        elif dataset_name == 'OfficeHome':
            domain = dataset(root=root, task=task, download=download, transform=transform)
            num_classes = domain.num_classes
        elif dataset_name == 'DomainNet':
            train_split = dataset(root=root, task=task, split='train', download=download, transform=transform)
            test_split = dataset(root=root, task=task, split='test', download=download, transform=transform)
            num_classes = train_split.num_classes
            domain = ConcatDataset([train_split, test_split])

        train_split, val_split = split_dataset(domain, int(len(domain) * split_ratio), seed)

        train_split_list.append(train_split)
        val_split_list.append(val_split)

    train_dataset = ConcatDataset(train_split_list)
    val_dataset = ConcatDataset(val_split_list)
    all_dataset = ConcatDataset([train_dataset, val_dataset])

    dataset_dict = {
        'train': train_dataset,
        'val': val_dataset,
        'all': all_dataset
    }
    return dataset_dict[split], num_classes


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n data points in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    idxes = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(idxes)
    subset_1 = idxes[:n]
    subset_2 = idxes[n:]
    return Subset(dataset, subset_1), Subset(dataset, subset_2)


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


def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=True,
                        random_gray_scale=True):
    """
    resizing mode:
        - default: random resized crop with scale factor(0.7, 1.0) size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
        - res.|crop: resize the image to 256 and take a random crop of size 224;
        - res.sma|crop: resize the image keeping its aspect ratio such that the
            smaller side is 256, then take a random crop of size 224;
        – inc.crop: “inception crop” from (Szegedy et al., 2015);
        – cif.crop: resize the image to 224, zero-pad it by 28 on each side, then take a random crop of size 224.
    """
    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.7, 1.0))
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'res.':
        transform = T.Resize(224)
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224)
        ])
    elif resizing == "res.sma|crop":
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'inc.crop':
        transform = T.RandomResizedCrop(224)
    elif resizing == 'cif.crop':
        transform = T.Compose([
            T.Resize((224, 224)),
            T.Pad(28),
            T.RandomCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
    if random_gray_scale:
        transforms.append(T.RandomGrayscale())
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default'):
    """
    resizing mode:
        - default: resize the image to 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
    """
    if resizing == 'default':
        transform = ResizeImage(224)
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
