"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader

import utils
from tllib.self_training.self_tuning import Classifier, SelfTuning
from tllib.vision.transforms import MultipleApply
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger

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
    strong_augment = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                               auto_augment=args.auto_augment,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std)
    train_transform = MultipleApply([strong_augment, strong_augment])
    val_transform = utils.get_val_transform(args.val_resizing, norm_mean=args.norm_mean, norm_std=args.norm_std)
    print('train_transform: ', train_transform)
    print('val_transform:', val_transform)
    labeled_train_dataset, unlabeled_train_dataset, val_dataset = \
        utils.get_dataset(args.data,
                          args.num_samples_per_class,
                          args.root, train_transform,
                          val_transform,
                          seed=args.seed)
    print("labeled_dataset_size: ", len(labeled_train_dataset))
    print('unlabeled_dataset_size: ', len(unlabeled_train_dataset))
    print("val_dataset_size: ", len(val_dataset))

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)
    unlabeled_train_iter = ForeverDataIterator(unlabeled_train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    num_classes = labeled_train_dataset.num_classes

    backbone_q = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier_q = Classifier(backbone_q, num_classes, projection_dim=args.projection_dim,
                              bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer,
                              finetune=args.finetune).to(device)
    print(classifier_q)

    backbone_k = utils.get_model(args.arch)
    classifier_k = Classifier(backbone_k, num_classes, projection_dim=args.projection_dim,
                              bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer).to(device)

    selftuning = SelfTuning(classifier_q, classifier_k, num_classes, K=args.K, m=args.m, T=args.T).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier_q.get_parameters(args.lr), args.lr, momentum=0.9, weight_decay=args.wd,
                    nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.lr_gamma)

    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier_q.load_state_dict(checkpoint)
        acc1, avg = utils.validate(val_loader, classifier_q, args, device, num_classes)
        print(acc1)
        return

    # start training
    best_acc1 = 0.0
    best_avg = 0.0
    for epoch in range(args.epochs):
        # print lr
        print(lr_scheduler.get_lr())

        # train for one epoch
        train(labeled_train_iter, unlabeled_train_iter, selftuning, optimizer, epoch, args)

        # update lr
        lr_scheduler.step()

        # evaluate on validation set
        acc1, avg = utils.validate(val_loader, classifier_q, args, device, num_classes)

        # remember best acc@1 and save checkpoint
        torch.save(classifier_q.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        best_avg = max(avg, best_avg)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    print('best_avg = {:3.1f}'.format(best_avg))
    logger.close()


def train(labeled_train_iter: ForeverDataIterator, unlabeled_train_iter: ForeverDataIterator, selftuning: SelfTuning,
          optimizer: SGD, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    pgc_losses_labeled = AverageMeter('Pgc Loss (Labeled Data)', ':3.2f')
    pgc_losses_unlabeled = AverageMeter('Pgc Loss (Unlabeled Data)', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, pgc_losses_labeled, pgc_losses_unlabeled, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # define loss functions
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)

    # switch to train mode
    selftuning.train()

    end = time.time()
    batch_size = args.batch_size
    for i in range(args.iters_per_epoch):
        (l_q, l_k), labels_l = next(labeled_train_iter)
        (u_q, u_k), _ = next(unlabeled_train_iter)

        l_q, l_k = l_q.to(device), l_k.to(device)
        u_q, u_k = u_q.to(device), u_k.to(device)
        labels_l = labels_l.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # clear grad
        optimizer.zero_grad()

        # compute output
        pgc_logits_labeled, pgc_labels_labeled, y_l = selftuning(l_q, l_k, labels_l)
        # cross entropy loss
        cls_loss = F.cross_entropy(y_l, labels_l)

        # pgc loss on labeled samples
        pgc_loss_labeled = criterion_kl(pgc_logits_labeled, pgc_labels_labeled)
        (cls_loss + pgc_loss_labeled).backward()

        # pgc loss on unlabeled samples
        _, y_pred = selftuning.encoder_q(u_q)
        _, pseudo_labels = torch.max(y_pred, dim=1)
        pgc_logits_unlabeled, pgc_labels_unlabeled, _ = selftuning(u_q, u_k, pseudo_labels)
        pgc_loss_unlabeled = criterion_kl(pgc_logits_unlabeled, pgc_labels_unlabeled)
        pgc_loss_unlabeled.backward()

        # compute gradient and do SGD step
        optimizer.step()

        # measure accuracy and record loss
        cls_losses.update(cls_loss.item(), batch_size)
        pgc_losses_labeled.update(pgc_loss_labeled.item(), batch_size)
        pgc_losses_unlabeled.update(pgc_loss_unlabeled.item(), batch_size)
        loss = cls_loss + pgc_loss_labeled + pgc_loss_unlabeled
        losses.update(loss.item(), batch_size)

        cls_acc = accuracy(y_l, labels_l)[0]
        cls_accs.update(cls_acc.item(), batch_size)

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
    parser.add_argument('--num-samples-per-class', default=4, type=int,
                        help='number of labeled samples per class')
    parser.add_argument('--train-resizing', default='default', type=str)
    parser.add_argument('--val-resizing', default='default', type=str)
    parser.add_argument('--norm-mean', default=(0.485, 0.456, 0.406), type=float, nargs='+',
                        help='normalization mean')
    parser.add_argument('--norm-std', default=(0.229, 0.224, 0.225), type=float, nargs='+',
                        help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=utils.get_model_names(),
                        help='backbone architecture: ' + ' | '.join(utils.get_model_names()) + ' (default: resnet50)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int,
                        help='dimension of bottleneck')
    parser.add_argument('--projection-dim', default=1024, type=int,
                        help='dimension of projection head')
    parser.add_argument('--no-pool', action='store_true', default=False,
                        help='no pool layer after the feature extractor')
    parser.add_argument('--pretrained-backbone', default=None, type=str,
                        help="pretrained checkpoint of the backbone "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='whether to use 10x smaller lr for backbone')
    # training parameters
    parser.add_argument('--T', default=0.07, type=float,
                        help="temperature (default: 0.07)")
    parser.add_argument('--K', default=32, type=int,
                        help="queue size (default: 32)")
    parser.add_argument('--m', default=0.999, type=float,
                        help="momentum coefficient (default: 0.999)")
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, metavar='LR', dest='lr',
                        help='initial learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--milestones', default=[12, 24, 36, 48], type=int, nargs='+',
                        help='epochs to decay learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default:5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run (default: 60)')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='number of iterations per epoch (default: 500)')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training ')
    parser.add_argument("--log", default='self_tuning', type=str,
                        help="where to save logs, checkpoints and debugging images")
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")
    args = parser.parse_args()
    main(args)
