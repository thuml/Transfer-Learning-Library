import math
import os
import random
import time
import warnings
import sys
import argparse
import shutil

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

sys.path.append('../../..')
from ftlib.finetune.delta import *
from common.modules.classifier import Classifier
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger

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
    train_transform = T.Compose([
        ResizeImage(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_dataset = dataset(root=args.root, split='train', sample_rate=args.sample_rate, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, split='test', sample_rate=100, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    train_iter = ForeverDataIterator(train_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    backbone_source = models.__dict__[args.arch](pretrained=True)
    num_classes = train_dataset.num_classes
    classifier = Classifier(backbone, num_classes).to(device)
    source_classifier = Classifier(backbone_source, head=backbone_source.copy_head(), num_classes=backbone_source.fc.out_features).to(device)
    for param in source_classifier.parameters():
        param.requires_grad = False
    source_classifier.eval()

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(args.lr), momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, gamma=args.lr_gamma)

    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1 = validate(val_loader, classifier, args)
        print(acc1)
        return

    # create intermediate layer getter
    if args.arch == 'resnet50':
        return_layers = ['backbone.layer1.2.conv3', 'backbone.layer2.3.conv3', 'backbone.layer3.5.conv3', 'backbone.layer4.2.conv3']
    elif args.arch == 'resnet101':
        return_layers = ['backbone.layer1.2.conv3', 'backbone.layer2.3.conv3', 'backbone.layer3.5.conv3', 'backbone.layer4.2.conv3']
    else:
        raise NotImplementedError(args.arch)
    source_getter = IntermediateLayerGetter(source_classifier, return_layers=return_layers)
    target_getter = IntermediateLayerGetter(classifier, return_layers=return_layers)

    # get regularization
    if args.regularization_type == 'l2_sp':
        backbone_regularization = SPRegularization(source_classifier.backbone, classifier.backbone)
    elif args.regularization_type == 'feature_map':
        backbone_regularization = BehavioralRegularization()
    elif args.regularization_type == 'attention_feature_map':
        attention_file = os.path.join(logger.root, args.attention_file)
        if not os.path.exists(attention_file):
            attention = calculate_channel_attention(train_dataset, return_layers, args)
            torch.save(attention, attention_file)
        else:
            print("Loading channel attention from", attention_file)
            attention = torch.load(attention_file)
            attention = [a.to(device) for a in attention]
        backbone_regularization = AttentionBehavioralRegularization(attention)
    else:
        raise NotImplementedError(args.regularization_type)

    head_regularization = L2Regularization(nn.ModuleList([classifier.head, classifier.bottleneck]))

    # start training
    best_acc1 = 0.0

    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_iter, classifier, backbone_regularization, head_regularization, target_getter, source_getter, optimizer, epoch, args)
        lr_scheduler.step()

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    logger.close()


def calculate_channel_attention(dataset, return_layers, args):
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = Classifier(backbone, dataset.num_classes).to(device)
    optimizer = SGD(classifier.get_parameters(args.lr), momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    data_loader = DataLoader(dataset, batch_size=args.attention_batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=False)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(math.log(0.1) / args.attention_lr_decay_epochs))
    criterion = nn.CrossEntropyLoss()

    channel_weights = []
    for layer_id, name in enumerate(return_layers):
        layer = get_attribute(classifier, name)
        layer_channel_weight = [0] * layer.out_channels
        channel_weights.append(layer_channel_weight)

    # train the classifier
    classifier.train()
    classifier.backbone.requires_grad = False
    print("Pretrain a classifier to calculate channel attention.")
    for epoch in range(args.attention_epochs):
        losses = AverageMeter('Loss', ':3.2f')
        cls_accs = AverageMeter('Cls Acc', ':3.1f')
        progress = ProgressMeter(
            len(data_loader),
            [losses, cls_accs],
            prefix="Epoch: [{}]".format(epoch))

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = classifier(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cls_acc = accuracy(outputs, labels)[0]

            losses.update(loss.item(), inputs.size(0))
            cls_accs.update(cls_acc.item(), inputs.size(0))

            if i % args.print_freq == 0:
                progress.display(i)
        lr_scheduler.step()

    # calculate the channel attention
    print('Calculating channel attention.')
    classifier.eval()
    if args.attention_iteration_limit > 0:
        total_iteration = min(len(data_loader), args.attention_iteration_limit)
    else:
        total_iteration = len(args.data_loader)

    progress = ProgressMeter(
        total_iteration,
        [],
        prefix="Iteration: ")

    for i, data in enumerate(data_loader):
        if i >= total_iteration:
            break
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs, _ = classifier(inputs)
        loss_0 = criterion(outputs, labels)
        progress.display(i)
        for layer_id, name in enumerate(tqdm(return_layers)):
            layer = get_attribute(classifier, name)
            for j in range(layer.out_channels):
                tmp = classifier.state_dict()[name + '.weight'][j,].clone()
                classifier.state_dict()[name + '.weight'][j,] = 0.0
                outputs, _ = classifier(inputs)
                loss_1 = criterion(outputs, labels)
                difference = loss_1 - loss_0
                difference = difference.detach().cpu().numpy().item()
                history_value = channel_weights[layer_id][j]
                channel_weights[layer_id][j] = 1.0 * (i * history_value + difference) / (i + 1)
                classifier.state_dict()[name + '.weight'][j, ] = tmp

    channel_attention = []
    for weight in channel_weights:
        weight = np.array(weight)
        weight = (weight - np.mean(weight)) / np.std(weight)
        weight = torch.from_numpy(weight).float().to(device)
        channel_attention.append(F.softmax(weight / 5).detach())
    return channel_attention


def train(train_iter: ForeverDataIterator, model: Classifier, backbone_regularization:nn.Module,  head_regularization:nn.Module,
          target_getter: IntermediateLayerGetter,
          source_getter: IntermediateLayerGetter,
          optimizer: SGD, epoch: int,  args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    losses_reg_head = AverageMeter('Loss (reg, head)', ':3.2f')
    losses_reg_backbone = AverageMeter('Loss (reg, backbone)', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, losses_reg_head, losses_reg_backbone, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x, labels = next(train_iter)
        x = x.to(device)
        label = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        intermediate_output_s, output_s = source_getter(x)
        intermediate_output_t, output_t = target_getter(x)
        y, f = output_t

        # measure accuracy and record loss
        cls_acc = accuracy(y, label)[0]
        cls_loss = F.cross_entropy(y, label)
        if args.regularization_type == 'feature_map':
            loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t)
        elif args.regularization_type == 'attention_feature_map':
            loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t)
        else:
            loss_reg_backbone = backbone_regularization()
        loss_reg_head = head_regularization()
        loss = cls_loss + args.trade_off_backbone * loss_reg_backbone + args.trade_off_head * loss_reg_head

        losses_reg_backbone.update(loss_reg_backbone.item() * args.trade_off_backbone, x.size(0))
        losses_reg_head.update(loss_reg_head.item() * args.trade_off_head, x.size(0))
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


def validate(val_loader: DataLoader, model: Classifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
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
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg


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

    parser = argparse.ArgumentParser(description='Delta for Finetuning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA',
                        help='dataset: ' + ' | '.join(dataset_names))
    parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet50)')
    parser.add_argument('--regularization-type', choices=['l2_sp', 'feature_map', 'attention_feature_map'],
                        default='attention_feature_map')
    parser.add_argument('--trade-off-backbone', default=0.01, type=float,
                         help='trade-off for backbone regularization')
    parser.add_argument('--trade-off-head', default=0.01, type=float,
                        help='trade-off for head regularization')

    # training parameters
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N',
                        help='mini-batch size (default: 48)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay-epochs', type=int, default=(12, ), nargs='+', help='epochs to decay lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                        metavar='W', help='weight decay (default: 0.0d)')
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
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    # parameters for calculating channel attention
    parser.add_argument("--attention-file", type=str, default='channel_attention.pt',
                        help="Where to save and load channel attention file.")
    parser.add_argument('--attention-batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size for calculating channel attention (default: 32)')
    parser.add_argument('--attention-epochs', default=10, type=int, metavar='N',
                        help='number of epochs to train for training before calculating channel weight')
    parser.add_argument('--attention-lr-decay-epochs', default=6, type=int, metavar='N',
                        help='epochs to decay lr for training before calculating channel weight')
    parser.add_argument('--attention-iteration-limit', default=10, type=int, metavar='N',
                        help='iteration limits for calculating channel attention, -1 means no limits')

    args = parser.parse_args()
    main(args)

