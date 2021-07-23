from typing import Tuple
import sys
import numpy as np
from torch.utils.data.dataset import Subset, ConcatDataset

sys.path.append('../../..')
import common.vision.datasets as datasets

supported_dataset = ['PACS', 'OfficeHome', 'DomainNet']


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
