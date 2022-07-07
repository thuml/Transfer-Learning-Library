"""
@author: Jiaxin Li
@contact: thulijx@gmail.com
"""
import argparse
import shutil
import time
import pprint

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wilds

import utils
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter


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
    # There are no well-developed data augmentation techniques for molecular graphs.
    train_transform = None
    val_transform = None
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_labeled_dataset, train_unlabeled_dataset, test_datasets, args.num_classes, args.class_names = \
        utils.get_dataset('ogb-molpcba', args.data_dir, args.unlabeled_list, args.test_list,
                          train_transform, val_transform, use_unlabeled=args.use_unlabeled, verbose=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = utils.get_model(args.arch, args.num_classes)
    model = model.cuda().to()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # Data loading code
    train_labeled_sampler = None
    train_labeled_loader = DataLoader(
        train_labeled_dataset, batch_size=args.batch_size[0], shuffle=(train_labeled_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_labeled_sampler,
        collate_fn=train_labeled_dataset.collate)

    # define loss function (criterion)
    criterion = utils.reduced_bce_logit_loss

    if args.phase == 'test':
        # resume from the latest checkpoint
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)
        for n, d in zip(args.test_list, test_datasets):
            print(n)
            utils.validate(d, model, -1, writer, args)
        return

    # start training
    best_val_metric = 0
    test_metric = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_labeled_loader, model, criterion, optimizer, epoch, writer, args)
        # evaluate on validation set
        for n, d in zip(args.test_list, test_datasets):
            print(n)
            if n == 'val':
                tmp_val_metric = utils.validate(d, model, epoch, writer, args)
            elif n == 'test':
                tmp_test_metric = utils.validate(d, model, epoch, writer, args)

        # remember best mse and save checkpoint
        is_best = tmp_val_metric > best_val_metric
        best_val_metric = max(tmp_val_metric, best_val_metric)
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
        if is_best:
            test_metric = tmp_test_metric
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))

    print("best val performance: {:.3f}".format(best_val_metric))
    print("test performance: {:.3f}".format(test_metric))
    logger.close()
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer, args):
    batch_time = AverageMeter('Time', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')

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
            losses.update(loss, input.size(0))
            global_step = epoch * len(train_loader) + i

            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            writer.add_scalar('train/loss', loss, global_step)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Loss {loss.val:.10f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader),
                args.batch_size[0] / batch_time.val,
                args.batch_size[0] / batch_time.avg,
                batch_time=batch_time, loss=losses))


if __name__ == '__main__':
    model_names = utils.get_model_names()
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='ogb-molpcba', choices=wilds.supported_datasets,
                        help='dataset: ' + ' | '.join(wilds.supported_datasets) +
                             ' (default: ogb-molpcba)')
    parser.add_argument('--unlabeled-list', nargs='+', default=[])
    parser.add_argument('--test-list', nargs='+', default=['val', 'test'])
    parser.add_argument('--metric', default='ap',
                        help='metric used to evaluate model performance. (default: average precision)')
    parser.add_argument('--use-unlabeled', action='store_true',
                        help='Whether use unlabeled data for training or not.')
    # model parameters
    parser.add_argument('--arch', '-a', metavar='ARCH', default='gin_virtual',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: gin_virtual)')
    # Learning rate schedule parameters
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Learning rate')
    parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                        metavar='W', help='weight decay (default: 0.0)')
    # training parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=(64, 64), type=int, nargs='+',
                        metavar='N', help='mini-batch size per process for source'
                                          ' and target domain (default: (64, 64))')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--log', type=str, default='src_only',
                        help='Where to save logs, checkpoints and debugging images.')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    args = parser.parse_args()
    main(args)
