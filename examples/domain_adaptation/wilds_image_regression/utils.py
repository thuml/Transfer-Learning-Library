import math
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import sys

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import wilds
import resnet_multispectral

sys.path.append('../../..')

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT


def get_model(arch, num_classes, num_channels):
    if arch == 'resnet18_ms':
        model = resnet_multispectral.ResNet18(num_classes=num_classes, num_channels=num_channels)
    elif arch == 'resnet34_ms':
        model = resnet_multispectral.ResNet34(num_classes=num_classes, num_channels=num_channels)
    elif arch == 'resnet50_ms':
        model = resnet_multispectral.ResNet50(num_classes=num_classes, num_channels=num_channels)
    elif arch == 'resnet101_ms':
        model = resnet_multispectral.ResNet101(num_classes=num_classes, num_channels=num_channels)
    elif arch == 'resnet152_ms':
        model = resnet_multispectral.ResNet152(num_classes=num_classes, num_channels=num_channels)
    else:
        raise ValueError('{} is not supported'.format(arch))

    return model


def get_dataset(dataset_name, root, unlabeled_list=("test_unlabeled",), test_list=("test",),
                split_scheme='official', transform_train=None, transform_test=None, verbose=True,
                **kwargs):
    labeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True, split_scheme=split_scheme, **kwargs)
    unlabeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True, unlabeled=True)
    num_classes = 1 if labeled_dataset.n_classes is None else labeled_dataset.n_classes
    train_labeled_dataset = labeled_dataset.get_subset("train", transform=transform_train)

    train_unlabeled_datasets = [
        unlabeled_dataset.get_subset(u, transform=transform_train)
        for u in unlabeled_list
    ]
    train_unlabeled_dataset = ConcatDataset(train_unlabeled_datasets)
    test_datasets = [
        labeled_dataset.get_subset(t, transform=transform_test)
        for t in test_list
    ]

    num_channels = labeled_dataset.get_input(0).size()[0]
    if verbose:
        print("Datasets")
        for n, d in zip(["train"] + unlabeled_list + test_list,
                        [train_labeled_dataset, ] + train_unlabeled_datasets + test_datasets):
            print("\t{}:{}".format(n, len(d)))

    return train_labeled_dataset, train_unlabeled_dataset, test_datasets, num_classes, num_channels


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

    # switch to evaluate mode
    model.eval()

    for input, target, metadata in tqdm.tqdm(val_loader):
        # compute output
        with torch.no_grad():
            output = model(input.cuda()).cpu()

        all_y_true.append(target)
        all_y_pred.append(output)
        all_metadata.append(metadata)

        sampled_inputs.append(input[0:1])
        sampled_targets.append(target[0:1])
        sampled_outputs.append(output[0:1])
        sampled_metadata.append(metadata[0:1])

    if args.local_rank == 0:

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
