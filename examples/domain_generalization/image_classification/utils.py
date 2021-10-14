"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import sys
import time
import timm
import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import Subset, ConcatDataset
import wilds

sys.path.append('../../..')
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter


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
    ) + wilds.supported_datasets


class ConcatDatasetWithDomainLabel(ConcatDataset):
    """ConcatDataset with domain label"""

    def __init__(self, *args, **kwargs):
        super(ConcatDatasetWithDomainLabel, self).__init__(*args, **kwargs)
        self.index_to_domain_id = {}
        domain_id = 0
        start = 0
        for end in self.cumulative_sizes:
            for idx in range(start, end):
                self.index_to_domain_id[idx] = domain_id
            start = end
            domain_id += 1

    def __getitem__(self, index):
        img, target = super(ConcatDatasetWithDomainLabel, self).__getitem__(index)
        domain_id = self.index_to_domain_id[index]
        return img, target, domain_id


def convert_from_wilds_dataset(dataset_name, wild_dataset):
    metadata_array = wild_dataset.metadata_array
    sample_idxes_per_domain = {}
    for idx, metadata in enumerate(metadata_array):
        if dataset_name == 'iwildcam':
            # In iwildcam dataset, domain id is specified by location
            domain = metadata[0].item()
        elif dataset_name == 'camelyon17':
            # In camelyon17 dataset, domain id is specified by hospital
            domain = metadata[0].item()
        elif dataset_name == 'fmow':
            # In fmow dataset, domain id is specified by (region, year) tuple
            domain = (metadata[0].item(), metadata[1].item())

        if domain not in sample_idxes_per_domain:
            sample_idxes_per_domain[domain] = []
        sample_idxes_per_domain[domain].append(idx)

    class Dataset:
        def __init__(self):
            self.dataset = wild_dataset

        def __getitem__(self, idx):
            x, y, metadata = self.dataset[idx]
            return x, y

        def __len__(self):
            return len(self.dataset)

    dataset = Dataset()
    concat_dataset = ConcatDatasetWithDomainLabel(
        [Subset(dataset, sample_idxes_per_domain[domain]) for domain in sample_idxes_per_domain])
    return concat_dataset


def get_dataset(dataset_name, root, task_list, split='train', download=True, transform=None, seed=0):
    assert split in ['train', 'val', 'test']
    if dataset_name in datasets.__dict__:
        # load datasets from common.vision.datasets
        # currently only PACS, OfficeHome and DomainNet are supported
        supported_dataset = ['PACS', 'OfficeHome', 'DomainNet']
        assert dataset_name in supported_dataset

        dataset = datasets.__dict__[dataset_name]

        train_split_list = []
        val_split_list = []
        test_split_list = []
        # we follow DomainBed and split each dataset randomly into two parts, with 80% samples and 20% samples
        # respectively, the former (larger) will be used as training set, and the latter will be used as validation set.
        split_ratio = 0.8
        num_classes = 0

        # under domain generalization setting, we use all samples in target domain as test set
        for task in task_list:
            if dataset_name == 'PACS':
                all_split = dataset(root=root, task=task, split='all', download=download, transform=transform)
                num_classes = all_split.num_classes
            elif dataset_name == 'OfficeHome':
                all_split = dataset(root=root, task=task, download=download, transform=transform)
                num_classes = all_split.num_classes
            elif dataset_name == 'DomainNet':
                train_split = dataset(root=root, task=task, split='train', download=download, transform=transform)
                test_split = dataset(root=root, task=task, split='test', download=download, transform=transform)
                num_classes = train_split.num_classes
                all_split = ConcatDataset([train_split, test_split])

            train_split, val_split = split_dataset(all_split, int(len(all_split) * split_ratio), seed)

            train_split_list.append(train_split)
            val_split_list.append(val_split)
            test_split_list.append(all_split)

        train_dataset = ConcatDatasetWithDomainLabel(train_split_list)
        val_dataset = ConcatDatasetWithDomainLabel(val_split_list)
        test_dataset = ConcatDatasetWithDomainLabel(test_split_list)

        dataset_dict = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        return dataset_dict[split], num_classes
    else:
        # load datasets from wilds
        # currently only iwildcam, camelyon17 and fmow are supported
        supported_dataset = ['iwildcam', 'camelyon17', 'fmow']
        assert dataset_name in supported_dataset

        dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
        num_classes = dataset.n_classes
        return convert_from_wilds_dataset(dataset_name,
                                          dataset.get_subset(split=split, transform=transform)), num_classes


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
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=True,
                        random_gray_scale=True):
    """
    resizing mode:
        - default: random resized crop with scale factor(0.7, 1.0) and size 224;
        - cen.crop: take the center crop of 224;
        - res.|cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
        - res2x: resize the image to 448;
        - res.|crop: resize the image to 256 and take a random crop of size 224;
        - res.sma|crop: resize the image keeping its aspect ratio such that the
            smaller side is 256, then take a random crop of size 224;
        – inc.crop: “inception crop” from (Szegedy et al., 2015);
        – cif.crop: resize the image to 224, zero-pad it by 28 on each side, then take a random crop of size 224.
    """
    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.7, 1.0))
    elif resizing == 'cen.crop':
        transform = T.CenterCrop(224)
    elif resizing == 'res.|cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'res':
        transform = ResizeImage(224)
    elif resizing == 'res2x':
        transform = ResizeImage(448)
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
        - res2x: resize the image to 448;
        - res.|cen.crop: resize the image to 256 and take the center crop of size 224;
    """
    if resizing == 'default':
        transform = ResizeImage(224)
    elif resizing == 'res2x':
        transform = ResizeImage(448)
    elif resizing == 'res.|cen.crop':
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


def collect_feature(data_loader, feature_extractor: nn.Module, device: torch.device,
                    max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features. This function is
    specific for domain generalization because each element in data_loader is a tuple
    (images, labels, domain_labels).

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, (images, target, domain_labels) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            images = images.to(device)
            feature = feature_extractor(images).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0)
