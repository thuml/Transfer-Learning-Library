import math
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import sys

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import wilds
from gnn import GINVirtual

sys.path.append('../../..')

def get_dataset(dataset_name, root, test_list=('test',),
                transform_train=None, transform_test=None, verbose=True):
    labeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
    num_classes = labeled_dataset.n_classes
    target_size = labeled_dataset.y_size
    train_labeled_dataset = labeled_dataset.get_subset('train', transform=transform_train)

    test_datasets = [
        labeled_dataset.get_subset(t, transform=transform_test)
        for t in test_list
    ]

    if dataset_name == 'fmow':
        from wilds.datasets.fmow_dataset import categories
        class_names = categories
    else:
        class_names = list(range(num_classes))

    if verbose:
        print('Datasets')
        for n, d in zip(['train'] + test_list,
                        [train_labeled_dataset, ] + test_datasets):
            print('\t{}:{}'.format(n, len(d)))
        print('\t#classes:', num_classes)

    return train_labeled_dataset, test_datasets, num_classes, class_names, target_size


def get_model(arch, num_classes):
    if arch == 'gin-virtual':
        model = GINVirtual(num_tasks=num_classes, dropout=0.5)
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
    if args.distributed:
        val_sampler = DistributedSampler(val_dataset)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size[0], shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=val_dataset.collate)

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