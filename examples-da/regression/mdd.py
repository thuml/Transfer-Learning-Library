import random
import time
import warnings
import sys
import argparse
import shutil
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

sys.path.append('../..')
from dalib.adaptation.mdd import RegressionMarginDisparityDiscrepancy as MarginDisparityDiscrepancy, ImageRegressor
import dalib.vision.datasets.regression as datasets
import dalib.vision.models as models
from dalib.utils.data import ForeverDataIterator
from dalib.utils.meter import AverageMeter, ProgressMeter
from dalib.utils.logger import CompleteLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)

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
    train_transform = T.Compose([
        T.Resize(128),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        T.Resize(128),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(root=args.root, task=args.source, split='train', download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, task=args.target, split='train', download=True, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, split='test', download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    num_factors = train_source_dataset.num_factors
    backbone = models.__dict__[args.arch](pretrained=True)
    regressor = ImageRegressor(backbone, num_factors, bottleneck_dim=args.bottleneck_dim, width=args.bottleneck_dim).to(
        device)
    mdd = MarginDisparityDiscrepancy(args.margin).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(regressor.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    if args.phase == 'test':
        regressor.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
        mae = validate(val_loader, regressor, args, train_source_dataset.factors)
        print(mae)
        return

    # start training
    best_mae = 100000.
    for epoch in range(args.epochs):
        # train for one epoch
        print("lr", lr_scheduler.get_lr())
        train(train_source_iter, train_target_iter, regressor, mdd, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        mae = validate(val_loader, regressor, args, train_source_dataset.factors)

        # remember best mae and save checkpoint
        torch.save(regressor.state_dict(), logger.get_checkpoint_path('latest'))
        if mae < best_mae:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_mae = min(mae, best_mae)
        print("mean MAE {:6.3f} best MAE {:6.3f}".format(mae, best_mae))

    print("best_mae = {:6.3f}".format(best_mae))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model, mdd: MarginDisparityDiscrepancy, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    source_losses = AverageMeter('Source Loss', ':6.3f')
    trans_losses = AverageMeter('Trans Loss', ':6.3f')
    mae_losses_s = AverageMeter('MAE Loss (s)', ':6.3f')
    mae_losses_t = AverageMeter('MAE Loss (t)', ':6.3f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, source_losses, trans_losses, mae_losses_s, mae_losses_t],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    mdd.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        optimizer.zero_grad()

        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device).float()
        x_t, labels_t = next(train_target_iter)
        x_t = x_t.to(device)
        labels_t = labels_t.to(device).float()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat([x_s, x_t], dim=0)
        outputs, outputs_adv = model(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        # compute mean square loss on source domain
        mse_loss = F.mse_loss(y_s, labels_s)

        # compute margin disparity discrepancy between domains
        transfer_loss = mdd(y_s, y_s_adv, y_t, y_t_adv)
        # for adversarial classifier, minimize negative mdd is equal to maximize mdd
        loss = mse_loss - transfer_loss * args.trade_off
        model.step()

        mae_loss_s = F.l1_loss(y_s, labels_s)
        mae_loss_t = F.l1_loss(y_t, labels_t)

        source_losses.update(mse_loss.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        mae_losses_s.update(mae_loss_s.item(), x_s.size(0))
        mae_losses_t.update(mae_loss_t.item(), x_s.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model, args: argparse.Namespace, factors) -> Tuple[float, float]:
    batch_time = AverageMeter('Time', ':6.3f')
    mae_losses = [AverageMeter('mae {}'.format(factor), ':6.3f') for factor in factors]
    progress = ProgressMeter(
        len(val_loader),
        [batch_time] + mae_losses,
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            for j in range(len(factors)):
                mae_loss = F.l1_loss(output[:, j], target[:, j])
                mae_losses[j].update(mae_loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        for i, factor in enumerate(factors):
            print("{} MAE {mae.avg:6.3f}".format(factor, mae=mae_losses[i]))
        mean_mae = sum(l.avg for l in mae_losses) / len(factors)
    return mean_mae


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='MDD for Regression Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='DSprites',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=512, type=int)
    parser.add_argument('--margin', type=float, default=1., help="margin gamma")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='mdd',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    args = parser.parse_args()
    print(args)
    main(args)

