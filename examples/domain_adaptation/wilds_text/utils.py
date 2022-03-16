import math
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import sys

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertTokenizerFast, DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, DistilBertModel

import wilds
sys.path.append('../../..')


class DistilBertClassifier(DistilBertForSequenceClassification):
    """
    Adapted from https://github.com/p-lambda/wilds
    """

    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


class DistilBertFeaturizer(DistilBertModel):
    """
    Adapted from https://github.com/p-lambda/wilds
    """

    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output


def get_transform(arch, max_token_length):
    """
    Adapted from https://github.com/p-lambda/wilds
    """
    if arch == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(arch)
    elif arch == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(arch)
    else:
        raise ValueError("Model: {arch} not recognized".format(arch))

    def transform(text):
        tokens = tokenizer(text, padding='max_length', truncation=True,
                           max_length=max_token_length, return_tensors='pt')
        if arch == 'bert-base-uncased':
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif arch == 'distilbert-base-uncased':
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform


def get_dataset(dataset_name, root, unlabeled_list=('extra_unlabeled',), test_list=('test',),
                transform_train=None, transform_test=None, verbose=True):
    labeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
    unlabeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True, unlabeled=True)
    num_classes = labeled_dataset.n_classes
    train_labeled_dataset = labeled_dataset.get_subset('train', transform=transform_train)

    train_unlabeled_datasets = [
        unlabeled_dataset.get_subset(u, transform=transform_train)
        for u in unlabeled_list
    ]
    train_unlabeled_dataset = ConcatDataset(train_unlabeled_datasets)
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
        for n, d in zip(['train'] + unlabeled_list + test_list,
                        [train_labeled_dataset, ] + train_unlabeled_datasets + test_datasets):
            print('\t{}:{}'.format(n, len(d)))
        print('\t#classes:', num_classes)

    return train_labeled_dataset, train_unlabeled_dataset, test_datasets, num_classes, class_names, labeled_dataset


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


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
        all_y_pred.append(output.argmax(1))
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