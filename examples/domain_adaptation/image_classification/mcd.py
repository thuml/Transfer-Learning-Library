"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp
from typing import Tuple

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.alignment.mcd import ImageClassifierHead, entropy, classifier_discrepancy
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

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
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    G = utils.get_model(args.arch, pretrain=not args.scratch).to(device)  # feature extractor
    # two image classifier heads
    pool_layer = nn.Identity() if args.no_pool else None
    F1 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim, pool_layer).to(device)
    F2 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim, pool_layer).to(device)

    # define optimizer
    # the learning rate is fixed according to origin paper
    optimizer_g = SGD(G.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = SGD([
        {"params": F1.parameters()},
        {"params": F2.parameters()},
    ], momentum=0.9, lr=args.lr, weight_decay=0.0005)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        G.load_state_dict(checkpoint['G'])
        F1.load_state_dict(checkpoint['F1'])
        F2.load_state_dict(checkpoint['F2'])

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(G, F1.pool_layer).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, G, F1, F2, args)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    best_results = None
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, G, F1, F2, optimizer_g, optimizer_f, epoch, args)

        # evaluate on validation set
        results = validate(val_loader, G, F1, F2, args)

        # remember best acc@1 and save checkpoint
        torch.save({
            'G': G.state_dict(),
            'F1': F1.state_dict(),
            'F2': F2.state_dict()
        }, logger.get_checkpoint_path('latest'))
        if max(results) > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = max(results)
            best_results = results

    print("best_acc1 = {:3.1f}, results = {}".format(best_acc1, best_results))

    # evaluate on test set
    checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    G.load_state_dict(checkpoint['G'])
    F1.load_state_dict(checkpoint['F1'])
    F2.load_state_dict(checkpoint['F2'])
    results = validate(test_loader, G, F1, F2, args)
    print("test_acc1 = {:3.1f}".format(max(results)))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          G: nn.Module, F1: ImageClassifierHead, F2: ImageClassifierHead,
          optimizer_g: SGD, optimizer_f: SGD, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    G.train()
    F1.train()
    F2.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        x = torch.cat((x_s, x_t), dim=0)
        assert x.requires_grad is False

        # measure data loading time
        data_time.update(time.time() - end)

        # Step A train all networks to minimize loss on source domain
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        g = G(x)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)

        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
               (entropy(y1_t) + entropy(y2_t)) * args.trade_off_entropy
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        # Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        g = G(x)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
               (entropy(y1_t) + entropy(y2_t)) * args.trade_off_entropy - \
               classifier_discrepancy(y1_t, y2_t) * args.trade_off
        loss.backward()
        optimizer_f.step()

        # Step C train genrator to minimize discrepancy
        for k in range(args.num_k):
            optimizer_g.zero_grad()
            g = G(x)
            y_1 = F1(g)
            y_2 = F2(g)
            y1_s, y1_t = y_1.chunk(2, dim=0)
            y2_s, y2_t = y_2.chunk(2, dim=0)
            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
            mcd_loss = classifier_discrepancy(y1_t, y2_t) * args.trade_off
            mcd_loss.backward()
            optimizer_g.step()

        cls_acc = accuracy(y1_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(mcd_loss.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead,
             F2: ImageClassifierHead, args: argparse.Namespace) -> Tuple[float, float]:
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

    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            g = G(images)
            y1, y2 = F1(g), F2(g)

            # measure accuracy and record loss
            acc1, = accuracy(y1, target)
            acc2, = accuracy(y2, target)
            if confmat:
                confmat.update(target, y1.argmax(1))
            top1_1.update(acc1.item(), images.size(0))
            top1_2.update(acc2.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc1 {top1_1.avg:.3f} Acc2 {top1_2.avg:.3f}'
              .format(top1_1=top1_1, top1_2=top1_2))
        if confmat:
            print(confmat.format(args.class_names))

    return top1_1.avg, top1_2.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCD for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trade-off-entropy', default=0.01, type=float,
                        help='the trade-off hyper-parameter for entropy loss')
    parser.add_argument('--num-k', type=int, default=4, metavar='K',
                        help='how many steps to repeat the generator update')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='mcd',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
