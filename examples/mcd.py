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
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')  # TODO remove this when published

from dalib.adaptation.mcd import ImageClassifierHead, entropy, classifier_discrepancy
import dalib.vision.datasets as datasets
import dalib.vision.models as models

from tools.utils import AverageMeter, ProgressMeter, accuracy, create_exp_dir, ForeverDataIterator
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
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_tranform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_tranform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    G = models.__dict__[args.arch](pretrained=True).cuda()  # feature extractor
    num_classes = train_source_dataset.num_classes
    # two image classifier heads
    F1 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim).cuda()
    F2 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim).cuda()

    # define optimizer
    # the learning rate is fixed according to origin paper
    optimizer_g = torch.optim.SGD(G.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = torch.optim.SGD(F1.get_parameters()+F2.get_parameters(), momentum=0.9, lr=args.lr, weight_decay=0.0005)

    G = torch.nn.DataParallel(G).cuda()
    F1 = torch.nn.DataParallel(F1).cuda()
    F2 = torch.nn.DataParallel(F2).cuda()

    # start training
    best_acc1 = 0.
    best_results = None
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, G, F1, F2, optimizer_g, optimizer_f, epoch, args)

        # evaluate on validation set
        results = validate(val_loader, G, F1, F2, args)

        # remember best acc@1 and save checkpoint
        if max(results) > best_acc1:
            best_acc1 = max(results)
            best_results = results

    print("best_acc1 = {:3.1f}, results = {}".format(best_acc1, best_results))


def train(train_source_iter, train_target_iter, G, F1, F2, optimizer_g, optimizer_f, epoch, args):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    G.train()
    F1.train()
    F2.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.cuda()
        x_t = x_t.cuda()
        labels_s = labels_s.cuda()
        labels_t = labels_t.cuda()

        # Step A train all networks to minimize loss on source domain
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        g_s, g_t = G(x_s), G(x_t)
        y1_s, y2_s = F1(g_s), F2(g_s)
        y1_t, y2_t = F1(g_t), F2(g_t)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
               0.01 * (entropy(y1_t) + entropy(y2_t))
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        # Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        g_s, g_t = G(x_s), G(x_t)
        y1_s, y2_s = F1(g_s), F2(g_s)
        y1_t, y2_t = F1(g_t), F2(g_t)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
               0.01 * (entropy(y1_t) + entropy(y2_t)) - classifier_discrepancy(y1_t, y2_t)
        loss.backward()
        optimizer_f.step()

        # Step C train genrator to minimize discrepancy
        for k in range(args.num_k):
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            g_t = G(x_t)
            y1_t, y2_t = F1(g_t), F2(g_t)
            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
            mcd_loss = classifier_discrepancy(y1_t, y2_t)
            mcd_loss.backward()
            optimizer_g.step()

        cls_acc = accuracy(y1_s, labels_s)[0]
        tgt_acc = accuracy(y1_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(mcd_loss.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, G, F1, F2, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1_1 = AverageMeter('Acc_1', ':6.2f')
    top1_2 = AverageMeter('Acc_2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1_1, top1_2],
        prefix='Test: ')

    # switch to evaluate mode
    G.eval()
    F1.eval()
    F2.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda()
            target = target.cuda()

            # compute output
            g = G(images)
            y1, y2 = F1(g), F2(g)

            # measure accuracy and record loss
            acc1, = accuracy(y1, target)
            acc2, = accuracy(y2, target)
            top1_1.update(acc1[0], images.size(0))
            top1_2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc1 {top1_1.avg:.3f} Acc2 {top1_2.avg:.3f}'
              .format(top1_1=top1_1, top1_2=top1_2))

    return top1_1.avg, top1_2.avg


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
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--num_k', type=int, default=4, metavar='K',
                        help='how many steps to repeat the generator update')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id(s) to use.')
    parser.add_argument('-i', '--iters_per_epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--bottleneck_dim', default=1024, type=int)

    args = parser.parse_args()
    # TODO remove this when published
    print(args)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)

