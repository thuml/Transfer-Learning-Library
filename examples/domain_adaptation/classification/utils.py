import copy
import sys
import time
import timm
import torch
import torch.nn.functional as F
import wilds

sys.path.append('../../..')
import common.vision.datasets as datasets
import common.vision.models as models
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_models(model_name):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=True)
        backbone.copy_head = lambda: copy.deepcopy(backbone.fc)
        backbone.out_features = backbone.fc.in_features
        backbone.reset_classifier(0, '')
    return backbone


def convert_from_wilds_dataset(wild_dataset):
    class Dataset:
        def __init__(self):
            self.dataset = wild_dataset

        def __getitem__(self, idx):
            x, y, metadata = self.dataset[idx]
            return x, y

        def __len__(self):
            return len(self.dataset)

    return Dataset()


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + wilds.supported_datasets


def get_dataset(dataset_name, root, source, target, train_transform, val_transform):
    if dataset_name in datasets.__dict__:
        # load datasets from common.vision.datasets
        dataset = datasets.__dict__[dataset_name]
        train_source_dataset = dataset(root=root, task=source, download=True, transform=train_transform)
        train_target_dataset = dataset(root=root, task=target, download=True, transform=train_transform)
        val_dataset = dataset(root=root, task=target, download=True, transform=val_transform)
        if dataset_name == 'DomainNet':
            test_dataset = dataset(root=root, task=target, split='test', download=True,
                                   transform=val_transform)
        else:
            test_dataset = val_dataset
        num_classes = dataset.num_classes
    else:
        # load datasets from common.vision.datasets
        dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
        num_classes = dataset.n_classes
        train_source_dataset = convert_from_wilds_dataset(dataset.get_subset('train', transform=train_transform))
        train_target_dataset = convert_from_wilds_dataset(dataset.get_subset('val', transform=train_transform))
        val_dataset = test_dataset = convert_from_wilds_dataset(dataset.get_subset('val', transform=val_transform))
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes


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
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        classes = confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
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

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(classes))

    return top1.avg

