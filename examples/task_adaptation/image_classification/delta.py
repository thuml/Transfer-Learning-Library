"""
@author: Yifei Ji, Junguang Jiang
@contact: jiyf990330@163.com, JiangJunguang1123@outlook.com
"""
import math
import os
import random
import time
import warnings
import sys
import argparse
import shutil

import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../../..')
from talib.finetune.delta import *
from common.modules.classifier import Classifier
from common.utils.data import ForeverDataIterator
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
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, not args.no_hflip, args.color_jitter)
    val_transform = utils.get_val_transform(args.val_resizing)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_dataset, val_dataset, num_classes = utils.get_dataset(args.data, args.root, train_transform,
                                                                val_transform, args.sample_rate, args.num_samples_per_classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    train_iter = ForeverDataIterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("training dataset size: {} test dataset size: {}".format(len(train_dataset), len(val_dataset)))

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, args.pretrained)
    backbone_source = utils.get_model(args.arch, args.pretrained)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = Classifier(backbone, num_classes, pool_layer=pool_layer, finetune=args.finetune).to(device)
    source_classifier = Classifier(backbone_source, num_classes=backbone_source.fc.out_features,
                                   head=backbone_source.copy_head(), pool_layer=pool_layer).to(device)
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
        acc1 = utils.validate(val_loader, classifier, args, device)
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
            attention = calculate_channel_attention(train_dataset, return_layers, num_classes, args)
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
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    logger.close()


def calculate_channel_attention(dataset, return_layers, num_classes, args):
    backbone = utils.get_model(args.arch)
    classifier = Classifier(backbone, num_classes).to(device)
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
        outputs = classifier(inputs)
        loss_0 = criterion(outputs, labels)
        progress.display(i)
        for layer_id, name in enumerate(tqdm(return_layers)):
            layer = get_attribute(classifier, name)
            for j in range(layer.out_channels):
                tmp = classifier.state_dict()[name + '.weight'][j,].clone()
                classifier.state_dict()[name + '.weight'][j,] = 0.0
                outputs = classifier(inputs)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delta for Finetuning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA')
    parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')
    parser.add_argument('-sc', '--num-samples-per-classes', default=None, type=int,
                        help='number of samples per classes.')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--color-jitter', action='store_true')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--finetune', action='store_true', help='whether use 10x smaller lr for backbone')
    parser.add_argument('--pretrained', default=None,
                        help="pretrained checkpoint of the backbone. "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
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
    parser.add_argument("--log", type=str, default='delta',
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

