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
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

import utils
from tllib.self_training.pi_model import L2ConsistencyLoss
from tllib.self_training.mean_teacher import EMATeacher
from tllib.self_training.self_ensemble import ClassBalanceLoss, ImageClassifier
from tllib.vision.transforms import ResizeImage, MultipleApply
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
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
    # we find self ensemble is sensitive to data augmentation. The following
    # data augmentation performs well for evaluated datasets
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        ResizeImage(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
        T.RandomGrayscale(),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target,
                          train_transform, val_transform, MultipleApply([train_transform, val_transform]))
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
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)

    # define optimizer and lr scheduler
    optimizer = Adam(classifier.get_parameters(), args.lr)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
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
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    if args.pretrain is None:
        # first pretrain the classifier wish source data
        print("Pretraining the model on source domain.")
        args.pretrain = logger.get_checkpoint_path('pretrain')
        pretrain_model = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                         pool_layer=pool_layer, finetune=not args.scratch).to(device)
        pretrain_optimizer = Adam(pretrain_model.get_parameters(), args.pretrain_lr)
        pretrain_lr_scheduler = LambdaLR(pretrain_optimizer,
                                         lambda x: args.pretrain_lr * (1. + args.lr_gamma * float(x)) ** (
                                             -args.lr_decay))

        # start pretraining
        for epoch in range(args.pretrain_epochs):
            # pretrain for one epoch
            utils.empirical_risk_minimization(train_source_iter, pretrain_model, pretrain_optimizer,
                                              pretrain_lr_scheduler, epoch, args,
                                              device)
            # validate to show pretrain process
            utils.validate(val_loader, pretrain_model, args, device)

        torch.save(pretrain_model.state_dict(), args.pretrain)
        print("Pretraining process is done.")

    checkpoint = torch.load(args.pretrain, map_location='cpu')
    classifier.load_state_dict(checkpoint)
    teacher = EMATeacher(classifier, alpha=args.alpha)
    consistency_loss = L2ConsistencyLoss().to(device)
    class_balance_loss = ClassBalanceLoss(num_classes).to(device)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, teacher, consistency_loss, class_balance_loss,
              optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          teacher: EMATeacher, consistency_loss, class_balance_loss,
          optimizer: Adam, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    cons_losses = AverageMeter('Cons Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, cons_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    teacher.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        (x_t1, x_t2), = next(train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t1 = x_t1.to(device)
        x_t2 = x_t2.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, _ = model(x_s)
        y_t, _ = model(x_t1)
        y_t_teacher, _ = teacher(x_t2)

        # classification loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        # compute output and mask
        p_t = F.softmax(y_t, dim=1)
        p_t_teacher = F.softmax(y_t_teacher, dim=1)
        confidence, _ = p_t_teacher.max(dim=1)
        mask = (confidence > args.threshold).float()

        # consistency loss
        cons_loss = consistency_loss(p_t, p_t_teacher, mask)
        # balance loss
        balance_loss = class_balance_loss(p_t) * mask.mean()

        loss = cls_loss + args.trade_off_cons * cons_loss + args.trade_off_balance * balance_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # update teacher
        teacher.update()

        # update statistics
        cls_acc = accuracy(y_s, labels_s)[0]
        cls_losses.update(cls_loss.item(), x_s.size(0))
        cons_losses.update(cons_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Ensemble for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
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
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--pretrain-lr', '--pretrain-learning-rate', default=3e-5, type=float,
                        help='initial pretrain learning rate', dest='pretrain_lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--alpha', default=0.99, type=float, help='ema decay rate (default: 0.99)')
    parser.add_argument('--threshold', default=0.8, type=float, help='confidence threshold')
    parser.add_argument('--trade-off-cons', default=3, type=float, help='trade off parameter for consistency loss')
    parser.add_argument('--trade-off-balance', default=0.01, type=float,
                        help='trade off parameter for class balance loss')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--pretrain-epochs', default=0, type=int, metavar='N',
                        help='number of total epochs(pretrain) to run')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='self_ensemble',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
