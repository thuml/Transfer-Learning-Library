"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
import tllib.vision.datasets.universal as datasets
from tllib.vision.datasets.universal import default_universal as universal
from tllib.modules.classifier import Classifier
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
    train_transform = utils.get_train_transform(args.train_resizing)
    val_transform = utils.get_val_transform(args.val_resizing)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    dataset = datasets.__dict__[args.data]
    source_dataset = universal(dataset, source=True)
    target_dataset = universal(dataset, source=False)
    train_source_dataset = source_dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_target_dataset = target_dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    val_dataset = target_dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    if args.data == 'DomainNet':
        test_dataset = target_dataset(root=args.root, task=args.target, split='test', download=True,
                                      transform=val_transform)
    else:
        test_dataset = val_dataset
    num_classes = train_source_dataset.num_classes
    num_common_classes = train_source_dataset.num_common_classes

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = Classifier(backbone, num_classes, pool_layer=pool_layer).to(device)

    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'))
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1, h_score = validate(test_loader, classifier, num_classes, num_common_classes, args)
        return

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # start training
    best_acc = 0.
    best_h_score = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, classifier, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc, h_score = validate(val_loader, classifier, num_classes, num_common_classes, args)
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))

        best_acc = max(acc, best_acc)
        if h_score > best_h_score:
            best_h_score = h_score
            # remember best h_score and save checkpoint
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))

    print('* Val Best Mean Acc@1 {:.3f}'.format(best_acc))
    print('* Val Best H-score {:.3f}'.format(best_h_score))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    test_acc, test_h_score = validate(test_loader, classifier, num_classes, num_common_classes, args)
    print('* Test Mean Acc@1 {:.3f} H-score {:.3f}'.format(test_acc, test_h_score))
    logger.close()


def train(train_source_iter: ForeverDataIterator, model: Classifier, optimizer: SGD, lr_scheduler: LambdaLR,
          epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    batch_size = args.batch_size
    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)

        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # compute output
        y_s, _ = model(x_s)
        loss = F.cross_entropy(y_s, labels_s)

        losses.update(loss.item(), batch_size)
        cls_acc = accuracy(y_s, labels_s)[0]
        cls_accs.update(cls_acc.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, num_classes, num_common_classes, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    unknown_class = num_classes
    confmat = ConfusionMatrix(num_classes=num_classes + 1)

    with torch.no_grad():
        end = time.time()
        for i, (images, label) in enumerate(val_loader):
            images = images.to(device)
            label = label.to(device)

            # compute output
            output = model(images)
            output = F.softmax(output, dim=1)
            confidence, prediction = output.max(dim=1)

            prediction[confidence < args.threshold] = unknown_class
            label[label >= unknown_class] = unknown_class
            confmat.update(label, prediction)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    _, accs, _ = confmat.compute()
    mean_acc = accs[accs != 0].mean().item() * 100
    known = accs[:num_common_classes].mean().item() * 100
    unknown = accs[-1].item() * 100
    h_score = 2 * known * unknown / (known + unknown)

    print('* Mean Acc@1 {:.3f}'.format(mean_acc))
    print('* Known Acc@1 {:.3f}'.format(known))
    print('* Unknown Acc@1 {:.3f}'.format(unknown))
    print('* H-score {:.3f}'.format(h_score))

    return mean_acc, h_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source Only for Universal Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain')
    parser.add_argument('-t', '--target', help='target domain')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    # training parameters
    parser.add_argument('--threshold', default=0.7, type=float,
                        help='When class confidence is less than the given threshold, '
                             'model will output "unknown" (default: 0.7)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run (default: 30)')
    parser.add_argument('-i', '--iters-per-epoch', default=200, type=int,
                        help='Number of iterations per epoch (default: 200)')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
