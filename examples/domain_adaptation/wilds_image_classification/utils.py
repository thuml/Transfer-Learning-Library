"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

import wilds
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
import timm

sys.path.append('../../..')

from tllib.vision.transforms import Denormalize
from tllib.utils.meter import AverageMeter, ProgressMeter


def get_model_names():
    return timm.list_models()


def get_model(model_name, pretrain=True):
    # load models from pytorch-image-models
    backbone = timm.create_model(model_name, pretrained=pretrain)
    try:
        backbone.out_features = backbone.get_classifier().in_features
        backbone.reset_classifier(0, '')
    except:
        backbone.out_features = backbone.head.in_features
        backbone.head = nn.Identity()
    return backbone


def get_dataset(dataset_name, root, unlabeled_list=("test_unlabeled",), test_list=("test",),
                transform_train=None, transform_test=None, verbose=True, transform_train_target=None):
    if transform_train_target is None:
        transform_train_target = transform_train
    labeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
    unlabeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True, unlabeled=True)
    num_classes = labeled_dataset.n_classes
    train_labeled_dataset = labeled_dataset.get_subset("train", transform=transform_train)

    train_unlabeled_datasets = [
        unlabeled_dataset.get_subset(u, transform=transform_train_target)
        for u in unlabeled_list
    ]
    train_unlabeled_dataset = ConcatDataset(train_unlabeled_datasets)
    test_datasets = [
        labeled_dataset.get_subset(t, transform=transform_test)
        for t in test_list
    ]

    if dataset_name == "fmow":
        from wilds.datasets.fmow_dataset import categories
        class_names = categories
    else:
        class_names = list(range(num_classes))

    if verbose:
        print("Datasets")
        for n, d in zip(["train"] + unlabeled_list + test_list,
                        [train_labeled_dataset, ] + train_unlabeled_datasets + test_datasets):
            print("\t{}:{}".format(n, len(d)))
        print("\t#classes:", num_classes)

    return train_labeled_dataset, train_unlabeled_dataset, test_datasets, num_classes, class_names


def collate_list(vec):
    """
    Adapted from https://github.com/p-lambda/wilds
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")


def get_train_transform(img_size, scale=None, ratio=None, hflip=0.5, vflip=0.,
                        color_jitter=0.4, auto_augment=None, interpolation='bilinear'):
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3. / 4., 4. / 3.))  # default imagenet ratio range
    transforms_list = [
        transforms.RandomResizedCrop(img_size, scale=scale, ratio=ratio, interpolation=_pil_interp(interpolation))]
    if hflip > 0.:
        transforms_list += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        transforms_list += [transforms.RandomVerticalFlip(p=vflip)]

    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            transforms_list += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            transforms_list += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            transforms_list += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        transforms_list += [transforms.ColorJitter(*color_jitter)]

    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return transforms.Compose(transforms_list)


def get_val_transform(img_size=224, crop_pct=None, interpolation='bilinear'):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    return transforms.Compose([
        transforms.Resize(scale_size, _pil_interp(interpolation)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def validate(val_dataset, model, epoch, writer, args):
    val_sampler = None
    if args.distributed:
        val_sampler = DistributedSampler(val_dataset)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size[0], shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    all_y_true = []
    all_y_pred = []
    all_metadata = []

    sampled_inputs = []
    sampled_outputs = []
    sampled_targets = []
    sampled_metadata = []

    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (input, target, metadata) in enumerate(val_loader):
        # compute output
        with torch.no_grad():
            output = model(input.cuda()).cpu()

        all_y_true.append(target)
        all_y_pred.append(output.argmax(1))
        all_metadata.append(metadata)

        sampled_inputs.append(input[0:1])
        sampled_targets.append(target[0:1])
        sampled_outputs.append(output[0:1])
        sampled_metadata.append(metadata[0:1])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            progress.display(i)

    if args.local_rank == 0:
        writer.add_figure(
            'test/predictions vs. actuals',
            plot_classes_preds(
                collate_list(sampled_inputs),
                collate_list(sampled_targets),
                collate_list(sampled_outputs),
                args.class_names,
                collate_list(sampled_metadata),
                val_dataset.metadata_map,
                nrows=min(int(len(val_loader) / 4), 50)
            ),
            global_step=epoch
        )

        # evaluate
        results = val_dataset.eval(
            collate_list(all_y_pred),
            collate_list(all_y_true),
            collate_list(all_metadata)
        )
        print(results[1])

        for k, v in results[0].items():
            if v == 0 or "Other" in k:
                continue
            writer.add_scalar("test/{}".format(k), v, global_step=epoch)

        return results[0][args.metric]


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def matplotlib_imshow(img):
    """helper function to show an image"""
    img = Denormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(img)
    img = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(img)


def plot_classes_preds(images, labels, outputs, class_names, metadata, metadata_map, nrows=4):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(outputs, 1)
    preds = np.squeeze(preds_tensor.numpy())
    probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, outputs)]

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, nrows * 4))
    domains = get_domain_names(metadata, metadata_map)
    for idx in np.arange(min(nrows * 4, len(images))):
        ax = fig.add_subplot(nrows, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2}\ndomain: {3})".format(
            class_names[preds[idx]],
            probs[idx] * 100.0,
            class_names[labels[idx]],
            domains[idx],
        ), color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def get_domain_names(metadata, metadata_map):
    return get_domain_ids(metadata)


def get_domain_ids(metadata):
    return [int(m[0]) for m in metadata]
