"""
@author: Jiaxin Li
@contact: thulijx@gmail.com
"""
import argparse
import os
import shutil
import time
import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import utils
from utils import Regressor
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter


def main(args):
    writer = None
    if args.local_rank == 0:
        logger = CompleteLogger(args.log, args.phase)
        if args.phase == 'train':
            writer = SummaryWriter(args.log)
        pprint.pprint(args)
        print("opt_level = {}".format(args.opt_level))
        print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
        print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

        print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    # Data loading code
    # Images in povertyMap dataset have 8 channels and traditional data augmentation
    # methods have no effect on performance.
    train_transform = None
    val_transform = None
    if args.local_rank == 0:
        print("train_transform: ", train_transform)
        print("val_transform: ", val_transform)

    train_labeled_dataset, train_unlabeled_dataset, test_datasets, args.num_channels = \
        utils.get_dataset('poverty', args.data_dir, args.unlabeled_list, args.test_list, args.split_scheme,
                          train_transform, val_transform, use_unlabeled=args.use_unlabeled,
                          verbose=args.local_rank == 0, fold=args.fold)

    # create model
    if args.local_rank == 0:
        print("=> creating model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, args.num_channels)
    pool_layer = nn.Identity() if args.no_pool else None
    model = Regressor(backbone, pool_layer=pool_layer, finetune=False)

    if args.sync_bn:
        import apex
        if args.local_rank == 0:
            print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda().to(memory_format=memory_format)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.gamma, step_size=args.step_size)

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # Data loading code
    train_labeled_sampler = None
    if args.distributed:
        train_labeled_sampler = DistributedSampler(train_labeled_dataset)

    train_labeled_loader = DataLoader(
        train_labeled_dataset, batch_size=args.batch_size[0], shuffle=(train_labeled_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_labeled_sampler)

    if args.phase == 'test':
        # resume from the latest checkpoint
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)
        for n, d in zip(args.test_list, test_datasets):
            if args.local_rank == 0:
                print(n)
            utils.validate(d, model, -1, writer, args)
        return

    # start training
    best_val_metric = 0
    test_metric = 0
    for epoch in range(args.epochs):
        if args.distributed:
            train_labeled_sampler.set_epoch(epoch)

        lr_scheduler.step(epoch)
        if args.local_rank == 0:
            print(lr_scheduler.get_last_lr())
            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[-1], epoch)
        # train for one epoch
        train(train_labeled_loader, model, optimizer, epoch, writer, args)
        # evaluate on validation set
        for n, d in zip(args.test_list, test_datasets):
            if args.local_rank == 0:
                print(n)
            if n == 'val':
                tmp_val_metric = utils.validate(d, model, epoch, writer, args)
            elif n == 'test':
                tmp_test_metric = utils.validate(d, model, epoch, writer, args)

        # remember best mse and save checkpoint
        if args.local_rank == 0:
            is_best = tmp_val_metric > best_val_metric
            best_val_metric = max(tmp_val_metric, best_val_metric)
            torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
            if is_best:
                test_metric = tmp_test_metric
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))

    print('best val performance: {:.3f}'.format(best_val_metric))
    print('test performance: {:.3f}'.format(test_metric))


def train(train_loader, model, optimizer, epoch, writer, args):
    batch_time = AverageMeter('Time', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target, metadata) in enumerate(train_loader):

        # compute output
        output, _ = model(input.cuda())
        loss = F.mse_loss(output, target.cuda())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            global_step = epoch * len(train_loader) + i

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                writer.add_scalar("train/loss", to_python_float(reduced_loss), global_step)

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})'.format(
                    epoch, i, len(train_loader),
                    args.world_size * args.batch_size[0] / batch_time.val,
                    args.world_size * args.batch_size[0] / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses))


if __name__ == '__main__':
    model_names = utils.get_model_names()
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('--unlabeled-list', nargs='+', default=[])
    parser.add_argument('--test-list', nargs='+', default=['val', 'test'])
    parser.add_argument('--metric', default='r_wg',
                        help='metric used to evaluate model performance.'
                             '(default: worst-U/R Pearson r)')
    parser.add_argument('--split-scheme', type=str,
                        help='Identifies how the train/val/test split is constructed.'
                             'Choices are dataset-specific.')
    parser.add_argument('--fold', type=str, default='A', choices=['A', 'B', 'C', 'D', 'E'],
                        help='Fold for poverty dataset. Poverty has 5 different cross validation folds,'
                             'each splitting the countries differently.')
    parser.add_argument('--use-unlabeled', action='store_true',
                        help='Whether use unlabeled data for training or not.')
    # model parameters
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18_ms',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18_ms)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    # Learning rate schedule parameters
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=int, default=0.96, help='parameter for StepLR scheduler')
    parser.add_argument('--step-size', type=int, default=1, help='parameter for StepLR scheduler')
    # training parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=(64, 64), type=int, nargs='+',
                        metavar='N', help='mini-batch size per process for source'
                                          ' and target domain (default: (64, 64))')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync-bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('--log', type=str, default='src_only',
                        help='Where to save logs, checkpoints and debugging images.')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    args = parser.parse_args()
    main(args)
