"""
@author: Jiaxin Li
@contact: thulijx@gmail.com
"""
import time
import sys

from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import wilds
import resnet_ms as models

sys.path.append('../../..')
from tllib.utils.meter import AverageMeter, ProgressMeter


class Regressor(nn.Module):
    """A generic Regressor class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any regressor head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the regressor or train from scratch. Default: True

    .. note::
        Different regressors are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Regressor` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Regressor` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this regressor is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Regressor.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: regressor's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_values`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, bottleneck: Optional[nn.Module] = None, bottleneck_dim: Optional[int] = -1,
                 head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(Regressor, self).__init__()
        self.backbone = backbone
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, 1)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


def get_dataset(dataset_name, root, unlabeled_list=("test_unlabeled",), test_list=("test",),
                split_scheme='official', transform_train=None, transform_test=None, use_unlabeled=True,
                verbose=True, **kwargs):
    labeled_dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True, split_scheme=split_scheme, **kwargs)
    train_labeled_dataset = labeled_dataset.get_subset("train", transform=transform_train)

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

    num_channels = labeled_dataset.get_input(0).size()[0]

    if verbose:
        print("Datasets")
        for n, d in zip(["train"] + unlabeled_list + test_list,
                        [train_labeled_dataset, ] + train_unlabeled_datasets + test_datasets):
            print("\t{}:{}".format(n, len(d)))

    return train_labeled_dataset, train_unlabeled_dataset, test_datasets, num_channels


def get_model_names():
    return sorted(name for name in models.__dict__ if
                  name.islower() and not name.startswith('__') and callable(models.__dict__[name]))


def get_model(arch, num_channels):
    if arch in models.__dict__:
        model = models.__dict__[arch](num_channels=num_channels)
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


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


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
