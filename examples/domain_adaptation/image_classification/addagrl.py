"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
Note: ADDA with gradient reverse layer
"""
import random
import time
import warnings
import sys
import copy
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

sys.path.append('../../..')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.adda import ImageClassifier
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.translation.cyclegan.util import set_requires_grad
from dalib.modules.grl import WarmStartGradientReverseLayer
from common.utils.data import ForeverDataIterator
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

sys.path.append('../classification')
import utils

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
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
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
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    source_classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                        pool_layer=pool_layer, finetune=not args.scratch).to(device)

    if args.phase == 'train' and args.pretrain is None:
        # first pretrain the classifier wish source data
        print("Pretraining the model on source domain.")
        args.pretrain = logger.get_checkpoint_path('pretrain')
        pretrain_model = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                         pool_layer=pool_layer, finetune=not args.scratch).to(device)
        pretrain_optimizer = SGD(pretrain_model.get_parameters(), args.pretrain_lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay, nesterov=True)
        pretrain_lr_scheduler = LambdaLR(pretrain_optimizer,
                                         lambda x: args.pretrain_lr * (1. + args.lr_gamma * float(x)) ** (
                                             -args.lr_decay))
        # start pretraining
        for epoch in range(args.pretrain_epochs):
            print("lr:", pretrain_lr_scheduler.get_lr())
            # pretrain for one epoch
            utils.pretrain(train_source_iter, pretrain_model, pretrain_optimizer, pretrain_lr_scheduler, epoch, args,
                           device)
            # validate to show pretrain process
            utils.validate(val_loader, pretrain_model, args, device)

        torch.save(pretrain_model.state_dict(), args.pretrain)
        print("Pretraining process is done.")

    checkpoint = torch.load(args.pretrain, map_location='cpu')
    source_classifier.load_state_dict(checkpoint)
    target_classifier = copy.deepcopy(source_classifier)

    # freeze source classifier
    set_requires_grad(source_classifier, False)
    source_classifier.freeze_bn()

    domain_discri = DomainDiscriminator(in_feature=source_classifier.features_dim, hidden_size=1024).to(device)

    # define loss function
    grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=2., max_iters=1000, auto_step=True)
    domain_adv = DomainAdversarialLoss(domain_discri, grl=grl).to(device)

    # define optimizer and lr scheduler
    # note that we only optimize target feature extractor
    optimizer = SGD(target_classifier.get_parameters(optimize_head=False) + domain_discri.get_parameters(), args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        target_classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(target_classifier.backbone, target_classifier.pool_layer,
                                          target_classifier.bottleneck).to(device)
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
        acc1 = utils.validate(test_loader, target_classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_source_iter, train_target_iter, source_classifier, target_classifier, domain_adv,
              optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, target_classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(target_classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    target_classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, target_classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          source_model: ImageClassifier, target_model: ImageClassifier, domain_adv: DomainAdversarialLoss,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses_transfer = AverageMeter('Transfer Loss', ':6.2f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_transfer, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    target_model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, _ = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        _, f_s = source_model(x_s)
        _, f_t = target_model(x_t)
        loss_transfer = domain_adv(f_s, f_t)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss_transfer.backward()
        optimizer.step()
        lr_scheduler.step()

        losses_transfer.update(loss_transfer.item(), x_s.size(0))
        domain_acc = domain_adv.domain_discriminator_accuracy
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADDA for Unsupervised Domain Adaptation')
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
    parser.add_argument('--pretrain', type=str, default=None,
                        help='pretrain checkpoint for classification model')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate of the classifier', dest='lr')
    parser.add_argument('--pretrain-lr', default=0.001, type=float, help='initial pretrain learning rate')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrain-epochs', default=3, type=int, metavar='N',
                        help='number of total epochs(pretrain) to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='addagrl',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
