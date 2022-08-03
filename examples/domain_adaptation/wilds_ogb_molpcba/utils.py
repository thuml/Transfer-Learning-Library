"""
@author: Jiaxin Li
@contact: thulijx@gmail.com
"""
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import wilds

sys.path.append('../../..')
import gin as models
from tllib.utils.meter import AverageMeter, ProgressMeter


def reduced_bce_logit_loss(y_pred, y_target):
    """
    Every item of y_target has n elements which may be labeled by nan.
    Nan values should not be used while calculating loss.
    So extract elements which are not nan first, and then calculate loss.
    """
    loss = nn.BCEWithLogitsLoss(reduction='none').cuda()
    is_labeled = ~torch.isnan(y_target)
    y_pred = y_pred[is_labeled].float()
    y_target = y_target[is_labeled].float()
    metrics = loss(y_pred, y_target)
    return metrics.mean()


def get_dataset(dataset_name, root, unlabeled_list=('test_unlabeled',), test_list=('test',),
                transform_train=None, transform_test=None, use_unlabeled=True, verbose=True):
    labeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
    train_labeled_dataset = labeled_dataset.get_subset('train', transform=transform_train)

    if use_unlabeled:
        unlabeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True, unlabeled=True)
        train_unlabeled_datasets = [
            unlabeled_dataset.get_subset(u, transform=transform_train)
            for u in unlabeled_list
        ]
        train_unlabeled_dataset = ConcatDataset(train_unlabeled_datasets)
    else:
        unlabeled_list = []
        train_unlabeled_datasets = []
        train_unlabeled_dataset = None

    test_datasets = [
        labeled_dataset.get_subset(t, transform=transform_test)
        for t in test_list
    ]

    if dataset_name == 'ogb-molpcba':
        num_classes = labeled_dataset.y_size
    else:
        num_classes = labeled_dataset.n_classes
    class_names = list(range(num_classes))

    if verbose:
        print('Datasets')
        for n, d in zip(['train'] + unlabeled_list + test_list,
                        [train_labeled_dataset, ] + train_unlabeled_datasets + test_datasets):
            print('\t{}:{}'.format(n, len(d)))
        print('\t#classes:', num_classes)

    return train_labeled_dataset, train_unlabeled_dataset, test_datasets, num_classes, class_names


def get_model_names():
    return sorted(name for name in models.__dict__ if
                  name.islower() and not name.startswith('__') and callable(models.__dict__[name]))


def get_model(arch, num_classes):
    if arch in models.__dict__:
        model = models.__dict__[arch](num_tasks=num_classes)
    else:
        raise ValueError('{} is not supported'.format(arch))
    return model


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
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size[0], shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=val_dataset.collate)

    all_y_true = []
    all_y_pred = []
    all_metadata = []

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
        all_y_pred.append(output)
        all_metadata.append(metadata)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            progress.display(i)

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
