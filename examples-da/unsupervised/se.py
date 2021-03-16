import random
import time
import warnings
import sys
import copy
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

sys.path.append('../..')
from dalib.adaptation.se import ema_model_update, ImageClassifier
from dalib.translation.cyclegan.util import set_requires_grad
import common.vision.datasets.selftraining as datasets
from common.vision.datasets.selftraining import perform_multiple_transforms as self_training_dataset
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

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
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
        T.RandomGrayscale(),
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
    st_dataset = self_training_dataset(dataset)
    train_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = st_dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, split='test', download=True,
                               transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    num_classes = train_source_dataset.num_classes
    backbone = models.__dict__[args.arch](pretrained=True)

    classifier = ImageClassifier(backbone, num_classes, args.bottleneck_dim).to(device)
    teacher = copy.deepcopy(classifier)

    # define optimizer and lr scheduler
    optimizer = Adam(classifier.get_parameters(), args.lr)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return

    if args.pretrain is None:
        # first pretrain the classifier wish source data
        print("Pretraining the model on source domain.")
        args.pretrain = logger.get_checkpoint_path('pretrain')
        pretrained_model = ImageClassifier(backbone, num_classes, args.bottleneck_dim).to(device)
        pretrain_optimizer = Adam(pretrained_model.get_parameters(), args.pretrain_lr)
        pretrain_lr_scheduler = LambdaLR(pretrain_optimizer,
                                         lambda x: args.pretrain_lr * (1. + args.lr_gamma * float(x)) ** (
                                             -args.lr_decay))

        # start pretraining
        for epoch in range(args.pretrain_epochs):
            # pretrain for one epoch
            pretrain(train_source_iter, pretrained_model, pretrain_optimizer, pretrain_lr_scheduler, epoch, args)
            # validate to show pretrain process
            acc1 = validate(val_loader, pretrained_model, args)

        torch.save(pretrained_model.state_dict(), args.pretrain)

        # show pretrain result
        pretrain_acc = validate(val_loader, pretrained_model, args)
        print("pretrain_acc1 = {:3.1f}".format(pretrain_acc))

    checkpoint = torch.load(args.pretrain, map_location='cpu')
    classifier.load_state_dict(checkpoint)
    teacher = copy.deepcopy(classifier)

    # start training
    set_requires_grad(teacher, False)
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, teacher, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def pretrain(train_source_iter: ForeverDataIterator, model: ImageClassifier, optimizer: Adam, lr_scheduler: LambdaLR,
             epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          teacher: ImageClassifier, optimizer: Adam, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    cons_losses = AverageMeter('Cons Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, cons_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    teacher.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t1, x_t2, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t1 = x_t1.to(device)
        x_t2 = x_t2.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, _ = model(x_s)
        y_t, _ = model(x_t1)
        y_t_teacher, _ = teacher(x_t2)

        # classification loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        # consistent loss
        y_t = F.softmax(y_t, dim=1)
        y_t_teacher = F.softmax(y_t_teacher, dim=1)

        cons_loss = ((y_t - y_t_teacher) ** 2).sum(dim=1)
        max_prob, _ = y_t_teacher.max(dim=1)
        mask = (max_prob > args.threshold).float()
        cons_loss = (cons_loss * mask).mean()

        # balance loss
        class_distribution = y_t.mean(dim=0)
        num_classes = y_t.shape[1]
        uniform_distribution = torch.ones(num_classes).to(device) / num_classes
        balance_loss = F.binary_cross_entropy(class_distribution, uniform_distribution)
        balance_loss = mask.mean() * balance_loss

        loss = cls_loss + args.trade_off_cons * cons_loss + args.trade_off_balance * balance_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # update teacher
        ema_model_update(model, teacher, args.alpha)

        # update statistics
        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]
        cls_losses.update(cls_loss.item(), x_s.size(0))
        cons_losses.update(cons_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
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
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

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
            if confmat:
                confmat.update(target, output.argmax(1))
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
        if confmat:
            print(confmat.format(classes))

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

    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
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
    parser.add_argument('--pretrain', type=str, default=None,
                        help='pretrain checkpoints for classification model')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--pretrain-lr', '--pretrain-learning-rate', default=3e-5, type=float,
                        help='initial pretrain learning rate', dest='pretrain_lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--alpha', default=0.99, type=float, help='ema decay rate (default: 0.99)')
    parser.add_argument('--threshold', default=0.8, type=float, help='confidence threshold')
    parser.add_argument('--trade-off-cons', default=3, type=float, help='trade off parameter for consistent loss')
    parser.add_argument('--trade-off-balance', default=0.01, type=float,
                        help='trade off parameter for class balance loss')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--pretrain-epochs', default=5, type=int, metavar='N',
                        help='number of total epochs(pretrain) to run')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
