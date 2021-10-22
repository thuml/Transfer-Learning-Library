"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import sys
import argparse
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as T

sys.path.append('../..')
from ssllib.rand_augment import RandAugment
from ssllib.self_tuning import Classifier, SelfTuning
from common.vision.transforms import ResizeImage, MultipleApply
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.data import ForeverDataIterator
from common.utils.logger import CompleteLogger

sys.path.append('.')
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
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    strong_augmentation = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.)),
        T.RandomHorizontalFlip(),
        RandAugment(n=2, m=10),
        T.ToTensor(),
        normalize
    ])

    train_transform = MultipleApply([strong_augmentation, strong_augmentation])
    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    labeled_train_dataset, unlabeled_train_dataset, val_dataset = utils.get_dataset(args.data, args.root,
                                                                                    args.sample_rate, train_transform,
                                                                                    val_transform)
    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True)
    unlabeled_train_iter = ForeverDataIterator(unlabeled_train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    num_classes = labeled_train_dataset.num_classes

    backbone_q = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier_q = Classifier(backbone_q, num_classes, projection_dim=args.projection_dim, pool_layer=pool_layer,
                              finetune=not args.scratch).to(device)
    classifier_q = nn.DataParallel(classifier_q)

    backbone_k = utils.get_model(args.arch)
    classifier_k = Classifier(backbone_k, num_classes, projection_dim=args.projection_dim, pool_layer=pool_layer).to(
        device)
    classifier_k = nn.DataParallel(classifier_k)

    selftuning = SelfTuning(classifier_q, classifier_k, num_classes, K=args.K, m=args.m, T=args.T).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier_q.module.get_parameters(args.lr), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.wd, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.lr_gamma)

    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier_q.load_state_dict(checkpoint)
        acc1 = utils.validate(val_loader, classifier_q, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        # print lr
        print(lr_scheduler.get_lr())

        # train for one epoch
        train(labeled_train_iter, unlabeled_train_iter, selftuning, optimizer, epoch, args)

        # update lr
        lr_scheduler.step()

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier_q, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier_q.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    logger.close()


def train(labeled_train_iter: ForeverDataIterator, unlabeled_train_iter: ForeverDataIterator, selftuning: SelfTuning,
          optimizer: SGD, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    pgc_losses_labeled = AverageMeter('Pgc Loss Labeled', ':3.2f')
    pgc_losses_unlabeled = AverageMeter('Pgc Loss Unlabeled', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, pgc_losses_labeled, pgc_losses_unlabeled, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # define loss functions
    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)

    # switch to train mode
    selftuning.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        (l_q, l_k), labels = next(labeled_train_iter)
        (u_q, u_k), _ = next(unlabeled_train_iter)

        l_q, l_k = l_q.to(device), l_k.to(device)
        u_q, u_k = u_q.to(device), u_k.to(device)
        labels = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output

        # labeled data
        pgc_logits_labeled, pgc_labels_labeled, y = selftuning(l_q, l_k, labels)
        # classification loss
        cls_loss = criterion_cls(y, labels)
        # pgc loss on labeled samples
        pgc_loss_labeled = criterion_kl(pgc_logits_labeled, pgc_labels_labeled)

        # unlabeled data
        _, y_pred = selftuning.encoder_q(u_q)
        _, pseudo_labels = torch.max(y_pred, dim=1)
        pgc_logits_unlabeled, pgc_labels_unlabeled, _ = selftuning(u_q, u_k, pseudo_labels)
        # pgc loss on unlabeled samples
        pgc_loss_unlabeled = criterion_kl(pgc_logits_unlabeled, pgc_labels_unlabeled)

        loss = cls_loss + pgc_loss_labeled + pgc_loss_unlabeled

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        cls_losses.update(cls_loss.item(), l_q.size(0))
        pgc_losses_labeled.update(pgc_loss_labeled.item(), l_q.size(0))
        pgc_losses_unlabeled.update(pgc_loss_unlabeled.item(), l_q.size(0))
        losses.update(loss.item(), l_q.size(0))

        cls_acc = accuracy(y, labels)[0]
        cls_accs.update(cls_acc.item(), l_q.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Tuning for Semi Supervised Learning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()))
    parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor. Used in models such as ViT.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--pretrained', default=None,
                        help="pretrained checkpoint of the backbone. "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument('--T', default=0.07, type=float, help="temperature. (default: 0.07)")
    parser.add_argument('--K', type=int, default=32, help="queue size. (default: 32)")
    parser.add_argument('--m', type=float, default=0.999, help="momentum coefficient. (default: 0.999)")
    parser.add_argument('--projection-dim', type=int, default=1024,
                        help="dimension of the projection head. (default: 1024)")
    # training parameters
    parser.add_argument('-b', '--batch-size', default=48, type=int, help='mini-batch size (default: 48)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('--milestones', type=int, default=[12, 24, 36, 48], nargs='+', help='epochs to decay lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='self_tuning',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    args = parser.parse_args()
    main(args)
