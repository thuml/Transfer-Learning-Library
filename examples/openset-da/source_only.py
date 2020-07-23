import random
import time
import warnings
import sys
import argparse
import copy
from typing import Sequence, List, Dict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')
from dalib.modules.classifier import Classifier
import dalib.vision.datasets as datasets
from dalib.vision.datasets.opensetda import default_openset as openset
import dalib.vision.models as models
from dalib.vision.transforms import ResizeImage
from dalib.utils.data import ForeverDataIterator
from dalib.utils.metric import accuracy, partial_accuracy
from dalib.utils.avgmeter import AverageMeter, ProgressMeter, ClassWiseAccuracyMeter
from dalib.optim.lr_scheduler import StepwiseLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super(ImageClassifier, self).__init__(backbone, num_classes)

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params



def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_tranform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    source_dataset = openset(dataset, source=True)
    target_dataset = openset(dataset, source=False)
    train_source_dataset = source_dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = target_dataset(root=args.root, task=args.target, download=True, transform=val_tranform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = target_dataset(root=args.root, task=args.target, split='test', download=True, transform=val_tranform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    num_classes = train_source_dataset.num_classes + 1
    classifier = ImageClassifier(backbone, num_classes).to(device)
    print(val_dataset.classes)
    print(len(val_dataset.classes))
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_sheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=0.0003, decay_rate=0.75)
    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print(lr_sheduler.get_lr())
        # train for one epoch
        train(train_source_iter, classifier, optimizer,
              lr_sheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args, val_dataset.classes)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(best_model)
    acc1 = validate(test_loader, classifier, args, val_dataset.classes)
    print("test_acc1 = {:3.1f}".format(acc1))


def train(train_source_iter: ForeverDataIterator, model: Classifier, optimizer: SGD,
          lr_sheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        if lr_sheduler is not None:
            lr_sheduler.step()

        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: Classifier, args: argparse.Namespace, classes: Sequence) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    accuracy_meter = ClassWiseAccuracyMeter(classes, ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, accuracy_meter],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            softmax_output = F.softmax(output, dim=1)
            softmax_output[:, -1] = args.threshold

            # measure accuracy and record loss
            accuracy_meter.update(softmax_output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        all_acc = accuracy_meter.average_accuracy(classes)
        print(' * All {all:.3f} Known {known:.3f} Unknown {unknown:.3f}'
                .format(all=all_acc, known=accuracy_meter.average_accuracy(classes[:-1]),
                        unknown=accuracy_meter.accuracy(classes[-1])))

    return all_acc


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='When class confidence is less than the given threshold, '
                             'model will output "unknown" (default: 0.5)')
    args = parser.parse_args()
    print(args)
    main(args)

