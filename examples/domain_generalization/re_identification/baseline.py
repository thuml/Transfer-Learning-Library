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
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append('../../..')
from common.vision.models.reid.loss import CrossEntropyLossWithLabelSmooth, SoftTripletLoss
from common.vision.models.reid.identifier import ReIdentifier
import common.vision.datasets.reid as datasets
from common.vision.datasets.reid.convert import convert_to_pytorch_dataset
from common.utils.scheduler import WarmupMultiStepLR
from common.utils.metric.reid import validate, visualize_ranked_results
from common.utils.data import ForeverDataIterator, RandomMultipleGallerySampler
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger

sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.height, args.width, args.train_resizing,
                                                random_horizontal_flip=True,
                                                random_color_jitter=False,
                                                random_gray_scale=False)
    val_transform = utils.get_val_transform(args.height, args.width)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    working_dir = osp.dirname(osp.abspath(__file__))
    root = osp.join(working_dir, args.root)

    # source dataset
    source_dataset = datasets.__dict__[args.source](root=osp.join(root, args.source.lower()))
    sampler = RandomMultipleGallerySampler(source_dataset.train, args.num_instances)
    train_loader = DataLoader(
        convert_to_pytorch_dataset(source_dataset.train, root=source_dataset.images_dir, transform=train_transform),
        batch_size=args.batch_size, num_workers=args.workers, sampler=sampler, pin_memory=True, drop_last=True)
    train_iter = ForeverDataIterator(train_loader)
    val_loader = DataLoader(
        convert_to_pytorch_dataset(list(set(source_dataset.query) | set(source_dataset.gallery)),
                                   root=source_dataset.images_dir,
                                   transform=val_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    # target dataset
    target_dataset = datasets.__dict__[args.target](root=osp.join(root, args.target.lower()))
    test_loader = DataLoader(
        convert_to_pytorch_dataset(list(set(target_dataset.query) | set(target_dataset.gallery)),
                                   root=target_dataset.images_dir,
                                   transform=val_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    # create model
    num_classes = source_dataset.num_train_pids
    backbone = utils.get_model(args.arch)
    pool_layer = nn.Identity() if args.no_pool else None
    model = ReIdentifier(backbone, num_classes, finetune=args.finetune, pool_layer=pool_layer).to(device)
    model = DataParallel(model)

    # define optimizer and learning rate scheduler
    optimizer = Adam(model.module.get_parameters(base_lr=args.lr, rate=args.rate), args.lr,
                     weight_decay=args.weight_decay)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.1,
                                     warmup_steps=args.warmup_steps)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # plot t-SNE
        utils.visualize_tsne(source_loader=val_loader, target_loader=test_loader, model=model,
                             filename=osp.join(logger.visualize_directory, 'analysis', 'TSNE.pdf'), device=device)
        # visualize ranked results
        visualize_ranked_results(test_loader, model, target_dataset.query, target_dataset.gallery, device,
                                 visualize_dir=logger.visualize_directory, width=args.width, height=args.height,
                                 rerank=args.rerank)
        return

    if args.phase == 'test':
        print("Test on source domain:")
        validate(val_loader, model, source_dataset.query, source_dataset.gallery, device, cmc_flag=True,
                 rerank=args.rerank)
        print("Test on target domain:")
        validate(test_loader, model, target_dataset.query, target_dataset.gallery, device, cmc_flag=True,
                 rerank=args.rerank)
        return

    # define loss function
    criterion_ce = CrossEntropyLossWithLabelSmooth(num_classes).to(device)
    criterion_triplet = SoftTripletLoss(margin=args.margin).to(device)

    # start training
    best_val_mAP = 0.
    best_test_mAP = 0.
    for epoch in range(args.epochs):
        # print learning rate
        print(lr_scheduler.get_lr())

        # train for one epoch
        train(train_iter, model, criterion_ce, criterion_triplet, optimizer, epoch, args)

        # update learning rate
        lr_scheduler.step()

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):

            # evaluate on validation set
            print("Validation on source domain...")
            _, val_mAP = validate(val_loader, model, source_dataset.query, source_dataset.gallery, device,
                                  cmc_flag=True)

            # remember best mAP and save checkpoint
            torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
            if val_mAP > best_val_mAP:
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_val_mAP = max(val_mAP, best_val_mAP)

            # evaluate on test set
            print("Test on target domain...")
            _, test_mAP = validate(test_loader, model, target_dataset.query, target_dataset.gallery, device,
                                   cmc_flag=True, rerank=args.rerank)
            best_test_mAP = max(test_mAP, best_test_mAP)

    # evaluate on test set
    model.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    print("Test on target domain:")
    _, test_mAP = validate(test_loader, model, target_dataset.query, target_dataset.gallery, device,
                           cmc_flag=True, rerank=args.rerank)
    print("test mAP on target = {}".format(test_mAP))
    print("oracle mAP on target = {}".format(best_test_mAP))
    logger.close()


def train(train_iter: ForeverDataIterator, model, criterion_ce: CrossEntropyLossWithLabelSmooth,
          criterion_triplet: SoftTripletLoss, optimizer: Adam, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_ce = AverageMeter('CeLoss', ':3.2f')
    losses_triplet = AverageMeter('TripletLoss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_ce, losses_triplet, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i in range(args.iters_per_epoch):
        x, _, labels, _ = next(train_iter)
        x = x.to(device)
        labels = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, f = model(x)

        # cross entropy loss
        loss_ce = criterion_ce(y, labels)
        # triplet loss
        loss_triplet = criterion_triplet(f, f, labels)
        loss = loss_ce + loss_triplet * args.trade_off

        cls_acc = accuracy(y, labels)[0]
        losses_ce.update(loss_ce.item(), x.size(0))
        losses_triplet.update(loss_triplet.item(), x.size(0))
        losses.update(loss.item(), x.size(0))
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )
    parser = argparse.ArgumentParser(description="Baseline for Domain Generalizable ReID")
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-s', '--source', type=str, help='source domain')
    parser.add_argument('-t', '--target', type=str, help='target domain')
    parser.add_argument('--train-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='reid_resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: reid_resnet50)')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--finetune', action='store_true', help='whether use 10x smaller lr for backbone')
    parser.add_argument('--rate', type=float, default=0.2)
    # training parameters
    parser.add_argument('--trade-off', type=float, default=1,
                        help='trade-off hyper parameter between cross entropy loss and triplet loss')
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--warmup-steps', type=int, default=10, help='number of warp-up steps')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    parser.add_argument('--eval-step', type=int, default=40)
    parser.add_argument('--iters-per-epoch', type=int, default=400)
    parser.add_argument('--print-freq', type=int, default=40)
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument('--rerank', action='store_true', help="evaluation only")
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
