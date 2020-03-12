import random
import time
import warnings
import sys
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')  # TODO remove this when published

import dalib.adaptation as adaptation
import dalib.vision.datasets as datasets
import dalib.vision.models as models

from tools.io_utils import AverageMeter, ProgressMeter, accuracy, create_exp_dir
from tools.transforms import ResizeImage


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    print("Use GPU: {} for training".format(args.gpu))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize_size = 256
    crop_size = 224
    train_transform = transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_tranform = transforms.Compose([
        ResizeImage(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

    train_source_dataset = datasets.__dict__[args.data](
        root=args.root, task=args.source, download=True, transform=train_transform)
    train_source_loader = torch.utils.data.DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, drop_last=True)

    train_target_dataset = datasets.__dict__[args.data](
        root=args.root, task=args.target, download=True, transform=train_transform)
    train_target_loader = torch.utils.data.DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, drop_last=True)
    val_dataset = datasets.__dict__[args.data](root=args.root, task=args.target, download=True, transform=val_tranform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    if args.iters_per_epoch is None:
        iters_per_epoch = max(len(train_source_loader), len(train_target_loader))
    else:
        iters_per_epoch = args.iters_per_epoch

    # create model
    cudnn.benchmark = True
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    num_classes = train_source_dataset.num_classes
    classifier = adaptation.mdd.Classifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, width=args.bottleneck_dim)
    mdd = adaptation.mdd.MarginDisparityDiscrepancy(args.margin)

    all_parameters = classifier.get_parameters()
    classifier = torch.nn.DataParallel(classifier).cuda()

    # define optimizer
    optimizer = torch.optim.SGD(all_parameters, args.lr, momentum=args.momentum,
                                weight_decay=args.wd, nesterov=True)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, mdd, optimizer, epoch, iters_per_epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))


def train(train_source_iter, train_target_iter, classifier, mdd, optimizer, epoch, iters_per_epoch, args):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    classifier.train()

    end = time.time()
    for i in range(iters_per_epoch):
        adjust_learning_rate(optimizer, i + iters_per_epoch * epoch, args)
        optimizer.zero_grad()

        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.cuda()
        x_t = x_t.cuda()
        labels_s = labels_s.cuda()
        labels_t = labels_t.cuda()

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        outputs, outputs_adv = classifier(x, keep_adv_output=True)
        y_s, y_t = outputs.narrow(0, 0, x_s.size(0)), outputs.narrow(0, x_s.size(0), x_t.size(0))
        y_s_adv, y_t_adv = outputs_adv.narrow(0, 0, x_s.size(0)), outputs_adv.narrow(0, x_s.size(0), x_t.size(0))
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = mdd(y_s, y_s_adv, y_t, y_t_adv)
        loss = cls_loss + transfer_loss

        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def adjust_learning_rate(optimizer, iter, args):
    """Sets the learning rate to the initial LR decayed each iterations"""
    lr = args.lr * (1. + 0.0002 * iter) ** (-0.75)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
    return lr


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


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
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', default=0.004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id(s) to use.')
    parser.add_argument('-i', '--iters_per_epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--bottleneck_dim', default=1024, type=int)
    parser.add_argument('--save', type=str, default=None, help='If not None, save the scripts to this directory')

    args = parser.parse_args()
    # # TODO remove this when published
    print(args)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.save is not None:
        create_exp_dir(args.save, ['dalib', 'examples', 'tools'])

    main(args)

