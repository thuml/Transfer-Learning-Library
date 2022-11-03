"""
@author: Jiaxin Li
@contact: thulijx@gmail.com
"""
import argparse
import shutil
import time
import pprint

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

import wilds
from wilds.common.grouper import CombinatorialGrouper

import utils
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter
from tllib.utils.metric import accuracy


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    writer = SummaryWriter(args.log)
    pprint.pprint(args)

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.set_printoptions(precision=10)

    # Data loading code
    train_transform = utils.get_transform(args.arch, args.max_token_length)
    val_transform = utils.get_transform(args.arch, args.max_token_length)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_labeled_dataset, train_unlabeled_dataset, test_datasets, labeled_dataset, args.num_classes, args.class_names = \
        utils.get_dataset(args.data, args.data_dir, args.unlabeled_list, args.test_list,
                          train_transform, val_transform, use_unlabeled=args.use_unlabeled, verbose=True)

    # create model
    print("=> using model '{}'".format(args.arch))
    model = utils.get_model(args.arch, args.num_classes)
    model = model.cuda().to()

    # Data loading code
    train_labeled_sampler = None
    if args.uniform_over_groups:
        train_grouper = CombinatorialGrouper(dataset=labeled_dataset, groupby_fields=args.groupby_fields)
        groups, group_counts = train_grouper.metadata_to_group(train_labeled_dataset.metadata_array, return_counts=True)
        group_weights = 1 / group_counts
        weights = group_weights[groups]
        train_labeled_sampler = WeightedRandomSampler(weights, len(train_labeled_dataset), replacement=True)

    train_labeled_loader = DataLoader(
        train_labeled_dataset, batch_size=args.batch_size[0], shuffle=(train_labeled_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_labeled_sampler
    )

    no_decay = ['bias', 'LayerNorm.weight']
    decay_params = []
    no_decay_params = []
    for names, params in model.named_parameters():
        if any(nd in names for nd in no_decay):
            no_decay_params.append(params)
        else:
            decay_params.append(params)
    params = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params, lr=args.lr)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_training_steps=len(train_labeled_loader) * args.epochs,
                                                   num_warmup_steps=0)
    lr_scheduler.step_every_batch = True
    lr_scheduler.use_metric = False

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.phase == 'test':
        # resume from the latest checkpoint
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)
        for n, d in zip(args.test_list, test_datasets):
            print(n)
            utils.validate(d, model, -1, writer, args)
        return

    best_val_metric = 0
    test_metric = 0
    for epoch in range(args.epochs):
        lr_scheduler.step(epoch)
        print(lr_scheduler.get_last_lr())
        writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[-1], epoch)
        # train for one epoch
        train(train_labeled_loader, model, criterion, optimizer, epoch, writer, args)
        # evaluate on validation set
        for n, d in zip(args.test_list, test_datasets):
            print(n)
            if n == 'val':
                tmp_val_metric = utils.validate(d, model, epoch, writer, args)
            elif n == 'test':
                tmp_test_metric = utils.validate(d, model, epoch, writer, args)

        # remember best prec@1 and save checkpoint
        is_best = tmp_val_metric > best_val_metric
        best_val_metric = max(tmp_val_metric, best_val_metric)
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
        if is_best:
            test_metric = tmp_test_metric
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))

    print('best val performance: {:.3f}'.format(best_val_metric))
    print('test performance: {:.3f}'.format(test_metric))
    logger.close()
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer, args):
    batch_time = AverageMeter('Time', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    top1 = AverageMeter('Top 1', ':3.1f')

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target, metadata) in enumerate(train_loader):

        # compute output
        output = model(input.cuda())
        loss = criterion(output, target.cuda())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, = accuracy(output.data, target.cuda(), topk=(1,))

            losses.update(loss, input.size(0))
            top1.update(prec1, input.size(0))
            global_step = epoch * len(train_loader) + i

            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            writer.add_scalar("train/loss", loss, global_step)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                args.batch_size[0] / batch_time.val,
                args.batch_size[0] / batch_time.avg,
                batch_time=batch_time,
                loss=losses, top1=top1))


if __name__ == '__main__':
    model_names = utils.get_model_names()
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='civilcomments', choices=wilds.supported_datasets,
                        help='dataset: ' + ' | '.join(wilds.supported_datasets) +
                             ' (default: civilcomments)')
    parser.add_argument('--unlabeled-list', nargs='+', default=[])
    parser.add_argument('--test-list', nargs='+', default=["val", "test"])
    parser.add_argument('--metric', default='acc_wg',
                        help='metric used to evaluate model performance. (default: worst group accuracy)')
    parser.add_argument('--uniform-over-groups', action='store_true',
                        help='sample examples such that batches are uniform over groups')
    parser.add_argument('--groupby-fields', nargs='+',
                        help='Group data by given fields. It means that items which have the same'
                             'values in those fields should be grouped.')
    parser.add_argument('--use-unlabeled', action='store_true',
                        help='Whether use unlabeled data for training or not.')
    # model parameters
    parser.add_argument('--arch', '-a', metavar='ARCH', default='distilbert-base-uncased',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: distilbert-base-uncased)')
    parser.add_argument('--max-token-length', type=int, default=300,
                        help='The maximum size of a sequence.')
    # Learning rate schedule parameters
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Learning rate.')
    parser.add_argument('--weight-decay', '--wd', default=0.01, type=float,
                        metavar='W', help='weight decay (default: 0.01)')
    # training parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=(16, 16), type=int, nargs='+',
                        metavar='N', help='mini-batch size per process for source'
                                          ' and target domain (default: (16, 16))')
    parser.add_argument('--print-freq', '-p', default=200, type=int,
                        metavar='N', help='print frequency (default: 200)')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--log', type=str, default='src_only',
                        help='Where to save logs, checkpoints and debugging images.')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis'm only analysis the model.")
    args = parser.parse_args()
    main(args)
