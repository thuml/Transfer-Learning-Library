import random
import time
import warnings
import sys
import argparse
import numpy as np
import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.append('.')
import dalib.vision.models.segmentation as models
import dalib.vision.datasets.segmentation as datasets
import dalib.vision.datasets.segmentation.transforms as T
from dalib.utils.data import ForeverDataIterator
from dalib.utils.metric import ConfusionMatrix
from dalib.utils.avgmeter import AverageMeter, ProgressMeter, Meter
from dalib.adaptation.segmentation.advent import Discriminator, DomainAdversarialEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    os.makedirs(args.snapshot_dir, exist_ok=True)
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
    image_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    target_dataset = datasets.__dict__[args.target]
    train_target_dataset = target_dataset(
        root=args.target_root,
        transforms=T.Compose([
            T.RandomResizedCrop(size=args.train_size, ratio=(2., 2.), scale=(0.5, 1.)),
            T.RandomHorizontalFlip(),
        ]),
        mean=image_mean
    )
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_target_dataset = target_dataset(
        root=args.target_root, split='val',
        transforms=T.Resize(image_size=args.test_input_size, label_size=args.test_output_size),
        mean=image_mean
    )
    val_target_loader = DataLoader(val_target_dataset, batch_size=1, shuffle=False, pin_memory=True)

    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(
        root=args.source_root,
        transforms=T.Compose([
            T.RandomResizedCrop(size=args.train_size, ratio=args.resize_ratio, scale=(0.5, 1.)),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomHorizontalFlip(),
        ]),
        mean=image_mean
    )
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    num_classes = train_source_dataset.num_classes
    model = models.__dict__[args.arch](num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(model.get_parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_d = Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.99))
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. - float(x) / args.epochs / args.iters_per_epoch) ** (args.lr_power))
    lr_scheduler_d = LambdaLR(optimizer_d, lambda x: (1. - float(x) / args.epochs / args.iters_per_epoch) ** (args.lr_power))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d'])
        args.start_epoch = checkpoint['epoch'] + 1

    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).to(device)
    dann = DomainAdversarialEntropyLoss(discriminator)
    interp_train = nn.Upsample(size=args.train_size[::-1], mode='bilinear', align_corners=True)
    interp_val = nn.Upsample(size=args.test_output_size[::-1], mode='bilinear', align_corners=True)

    if args.test_only:
        confmat = validate(val_target_loader, model, interp_val, criterion, args)
        print(confmat)
        return

    # start training
    best_iou = 0.
    for epoch in range(args.start_epoch, args.epochs):
        print(lr_scheduler.get_lr(), lr_scheduler_d.get_lr())
        # train for one epoch
        train(train_source_iter, train_target_iter, model, interp_train, criterion, dann, optimizer,
              lr_scheduler, optimizer_d, lr_scheduler_d, epoch, args)

        # evaluate on validation set
        confmat = validate(val_target_loader, model, interp_val, criterion, args)
        print(confmat.format(train_source_dataset.classes))
        acc_global, acc, iu = confmat.compute()

        # calculate the mean iou over partial classes
        indexes = [train_source_dataset.classes.index(name) for name in train_source_dataset.EVALUATE_CLASSES]
        iu = iu[indexes]
        mean_iou = iu.mean()

        # remember best acc@1 and save checkpoint
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'best.pth'))
        print("Target: {} Best: {}".format(mean_iou, best_iou))

        torch.save(
            {
                'model': model.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'lr_scheduler_d': lr_scheduler_d.state_dict(),
                'epoch': epoch,
                'args': args
            },
            os.path.join(args.snapshot_dir, str(epoch) + '.pth')
        )


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model, interp, criterion, dann,
          optimizer: SGD, lr_scheduler: LambdaLR, optimizer_d: SGD, lr_scheduler_d: LambdaLR,
          epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_s = AverageMeter('Loss (s)', ':3.2f')
    losses_transfer = AverageMeter('Loss (transfer)', ':3.2f')
    losses_discriminator = AverageMeter('Loss (discriminator)', ':3.2f')
    accuracies_s = Meter('Acc (s)', ':3.2f')
    accuracies_t = Meter('Acc (t)', ':3.2f')
    iou_s = Meter('IoU (s)', ':3.2f')
    iou_t = Meter('IoU (t)', ':3.2f')

    confmat_s = ConfusionMatrix(model.num_classes)
    confmat_t = ConfusionMatrix(model.num_classes)
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_s, losses_transfer, losses_discriminator,
         accuracies_s, accuracies_t, iou_s, iou_t],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        labels_s = labels_s.long().to(device)
        x_t = x_t.to(device)
        labels_t = labels_t.long().to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        optimizer_d.zero_grad()

        # Step 1: Train the segmentation network, freeze the discriminator
        dann.eval()
        y_s = model(x_s)
        pred_s = interp(y_s)
        loss_cls_s = criterion(pred_s, labels_s)
        loss_cls_s.backward()

        # adversarial training to fool the discriminator
        y_t = model(x_t)
        pred_t = interp(y_t)
        loss_transfer = dann(pred_t, 'source')
        (loss_transfer * args.trade_off).backward()

        # Step 2: Train the discriminator
        dann.train()
        loss_discriminator = 0.5 * (dann(pred_s.detach(), 'source') + dann(pred_t.detach(), 'target'))
        loss_discriminator.backward()

        # compute gradient and do SGD step
        optimizer.step()
        optimizer_d.step()
        lr_scheduler.step()
        lr_scheduler_d.step()

        confmat_s.update(labels_s.flatten(), pred_s.argmax(1).flatten())
        confmat_t.update(labels_t.flatten(), pred_t.argmax(1).flatten())

        losses_s.update(loss_cls_s.item(), x_s.size(0))
        losses_transfer.update(loss_transfer.item(), x_s.size(0))
        losses_discriminator.update(loss_discriminator.item(), x_s.size(0))

        acc_global_s, acc_s, iu_s = confmat_s.compute()
        acc_global_t, acc_t, iu_t = confmat_t.compute()
        accuracies_s.update(acc_s.mean().item())
        accuracies_t.update(acc_t.mean().item())
        iou_s.update(iu_s.mean().item())
        iou_t.update(iu_t.mean().item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model, interp, criterion, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = Meter('Acc', ':3.2f')
    iou = Meter('IoU', ':3.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc, iou],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    confmat = ConfusionMatrix(model.num_classes)

    with torch.no_grad():
        end = time.time()
        for i, (x, label) in enumerate(val_loader):
            x = x.to(device)
            label = label.long().to(device)
            # compute output
            output = interp(model(x))
            loss = criterion(output, label)
            # measure accuracy and record loss
            confmat.update(label.flatten(), output.argmax(1).flatten())
            losses.update(loss.item(), x.size(0))

            acc_global, accs, iu = confmat.compute()
            acc.update(accs.mean().item())
            iou.update(iu.mean().item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return confmat


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

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('source_root', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='deeplabv2_resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: deeplabv2_resnet101)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N',
                        help='mini-batch size (default: 2)')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--lr-power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate (only for deeplab).")
    parser.add_argument("--lr-d", default=1e-4, type=float,
                        metavar='LR', help='initial learning rate for discriminator')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--snapshot-dir", type=str, default='snapshots',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument('--trade-off', type=float, default=0.001)
    parser.add_argument('--resize-ratio', nargs='+', type=float, default=(1.5, 8/3.),
                        help='the resize ratio for the random resize crop')
    parser.add_argument('--train-size', nargs='+', type=int, default=(1024, 512),
                        help='the input and output image size during training')
    parser.add_argument('--test-input-size', nargs='+', type=int, default=(1024, 512),
                        help='the input image size during test')
    parser.add_argument('--test-output-size', nargs='+', type=int, default=(2048, 1024),
                        help='the output image size during test')
    args = parser.parse_args()
    print(args)
    main(args)
