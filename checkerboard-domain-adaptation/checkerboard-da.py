from common.utils.analysis import collect_feature_and_labels, tsne, a_distance
from common.utils.logger import CompleteLogger
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.data import ForeverDataIterator
from common.vision.transforms import ResizeImage
import common.vision.models as models
from common.vision.datasets.checkerboard_officehome import CheckerboardOfficeHome
# from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mdann import MultidomainAdversarialLoss, ImageClassifier
# from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.multidomain_discriminator import MultidomainDiscriminator
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../..')

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
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if args.center_crop:
        train_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), normalize
        ])
    else:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), normalize
        ])
    val_transform = T.Compose(
        [ResizeImage(256),
         T.CenterCrop(224),
         T.ToTensor(), normalize])

    # assuming that args.sources and arg.targets are disjoint sets
    # num_domains = len(args.sources) + len(args.targets)

    # TODO: create the train, val, test, novel dataset
    transforms_list = [train_transform,
                       val_transform, val_transform, val_transform]

    datasets = CheckerboardOfficeHome(root=args.root,
                                      download=False,
                                      transforms=transforms_list)
    # display the category-style matrix
    print(datasets)

    # train_dataset = CheckerboardOfficeHome(root=args.root,
    #                                        download=True,
    #                                        dataset_type='train',
    #                                        transform=train_transform)
    train_loader = DataLoader(datasets.train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              drop_last=True)

    # val_dataset = CheckerboardOfficeHome(root=args.root,
    #                                  tasks=args.targets,
    #                                  download=True,
    #                                  dataset_type='val',
    #                                  transform=val_transform)
    val_loader = DataLoader(datasets.val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers)

    # test_dataset = CheckerboardOfficeHome(root=args.root,
    #                                       tasks=args.targets,
    #                                       download=True,
    #                                       dataset_type='test',
    #                                       transform=val_transform)
    test_loader = DataLoader(datasets.test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.workers,
                             drop_last=True)
    # novel_dataset = CheckerboardOfficeHome(root=args.root,
    #                                       tasks=args.targets,
    #                                       download=True,
    #                                       dataset_type='novel',
    #                                       transform=val_transform)
    novel_loader = DataLoader(datasets.novel_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              drop_last=True)

    train_iter = ForeverDataIterator(train_loader)
    val_iter = ForeverDataIterator(val_loader)  # train_target_iter

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(backbone,
                                 len(datasets.classes()),
                                 bottleneck_dim=args.bottleneck_dim).to(device)

    multidomain_discri = MultidomainDiscriminator(
        in_feature=classifier.features_dim,
        hidden_size=1024,
        num_domains=len(datasets.domains())).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() +
                    multidomain_discri.get_parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(
        optimizer, lambda x: args.lr *
        (1. + args.lr_gamma * float(x))**(-args.lr_decay))

    # define loss function
    multidomain_adv = MultidomainAdversarialLoss(multidomain_discri).to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'),
                                map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone,
                                          classifier.bottleneck).to(device)
        train_feature, train_labels = collect_feature_and_labels(
            train_loader, feature_extractor, device)
        val_feature, val_labels = collect_feature_and_labels(
            train_loader, feature_extractor, device)
        test_feature, test_labels = collect_feature_and_labels(
            train_loader, feature_extractor, device)
        source_feature = torch.cat(
            [train_feature, val_feature, test_feature], axis=0)
        source_labels = torch.cat(
            [train_labels, val_labels, test_labels], axis=0)
        novel_feature, novel_labels = collect_feature_and_labels(
            novel_loader, feature_extractor, device)
        source_domain_labels = CheckerboardOfficeHome.get_category(
            source_labels)
        novel_domain_labels = CheckerboardOfficeHome.get_category(
            novel_labels, classifier.num_classes)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature,
                       novel_feature,
                       filename=tSNE_filename,
                       source_domain_labels=source_domain_labels,
                       target_domain_labels=novel_domain_labels,
                       num_domains=len(datasets.domains()))
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        # A_distance = a_distance.calculate(source_feature, target_feature,
        #                                   device)
        # print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args, 'Test', 0)
        print(acc1)
        return

    if args.phase == 'novel':
        acc1 = validate(novel_loader, classifier, args, 'Novel', 0)
        print(acc1)
        return

    # name of tensorboard
    comment = f' Checkerboard OfficeHome arch={args.arch} seed={args.seed} epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} lr_gamma={args.lr_gamma} lr_decay={args.lr_decay} trade_off={args.trade_off}'
    # Summary writer for tensorboard
    tb = SummaryWriter(comment=comment)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_iter, classifier, multidomain_adv, optimizer,
              lr_scheduler, epoch, args, tb)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args, 'Validation', epoch, tb)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args, 'Test')
    print("test_acc1 = {:3.1f}".format(acc1))
    acc1 = validate(novel_loader, 'Novel', args, classifier)
    print("novel_acc1 = {:3.1f}".format(acc1))

    logger.close()
    tb.close()


def train(train_iter: ForeverDataIterator, model: ImageClassifier,
          multidomain_adv: MultidomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace,
          tb: SummaryWriter):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_losses = AverageMeter('Cls Loss', ':6.2f')
    transfer_losses = AverageMeter('Transfer Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # define number of classes to predict

    # switch to train mode
    model.train()
    multidomain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_tr, labels_tr = next(train_iter)

        # retrieve the class and domain from the modified office_home dataset
        class_labels_tr = CheckerboardOfficeHome.get_category(labels_tr)
        domain_labels_tr = CheckerboardOfficeHome.get_style(labels_tr)

        x_tr = x_tr.to(device)

        # add new labels to device
        class_labels_tr = class_labels_tr.to(device)
        domain_labels_tr = domain_labels_tr.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_tr, f_tr = model(x_tr)

        # Updating the loss functions with new labels
        # cls_loss = F.cross_entropy(y_s, labels_s)
        cls_loss = F.cross_entropy(y_tr, class_labels_tr)

        transfer_loss = multidomain_adv(f_tr, domain_labels_tr)
        domain_acc = multidomain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_tr, class_labels_tr)[0]

        losses.update(loss.item(), x_tr.size(0))
        cls_accs.update(cls_acc.item(), x_tr.size(0))
        domain_accs.update(domain_acc.item(), x_tr.size(0))
        # seperated loss between two heads
        cls_losses.update(cls_loss.item(), x_tr.size(0))
        transfer_losses.update(transfer_loss.item(), x_tr.size(0))

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

    # tensorboard updates
    tb.add_scalar('Category Classification Loss (Training Set)',
                  cls_losses.sum, epoch)
    tb.add_scalar('Style Discrimination Loss (Training Set)',
                  transfer_losses.sum, epoch)
    tb.add_scalar('Total Loss (Training Set)', losses.sum, epoch)
    tb.add_scalar(
        'Category Classification Accuracy (Training Set)', cls_accs.avg, epoch)
    tb.add_scalar(
        'Style Discrimination Accuracy (Training Set', domain_accs.avg, epoch)
    tb.add_histogram('Classification Softmax', y_tr, epoch)


def validate(val_loader: DataLoader,
             model: ImageClassifier,
             args: argparse.Namespace,
             dataset_type: str,
             epoch: Optional[int] = 0,
             tb: SummaryWriter = None) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
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
            target = CheckerboardOfficeHome.get_category(target)
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

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))
        if confmat:
            print(confmat.format(classes))
    # tensorboard update(s)
    if tb:
        tb.add_scalar(
            f'Classification Loss ({dataset_type} Set)', losses.sum, epoch)
        tb.add_scalar(
            f'Classification Accuracy ({dataset_type} Set)', top1.avg, epoch)

    return top1.avg


if __name__ == '__main__':
    architecture_names = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(
        description='DANN for Checkerboard Domain Adapation on the Office-Home Dataset')
    # dataset parameters
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    # parser.add_argument('-s', '--source', help='source domain(s)')
    # parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--center-crop',
                        default=False,
                        action='store_true',
                        help='whether use center crop during training')
    # model parameters
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                        ' | '.join(architecture_names) +
                        ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim',
                        default=256,
                        type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--trade-off',
                        default=1.,
                        type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b',
                        '--batch-size',
                        default=32,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.01,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--lr-gamma',
                        default=0.001,
                        type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--lr-decay',
                        default=0.75,
                        type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-3,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j',
                        '--workers',
                        default=2,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',
                        default=20,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i',
                        '--iters-per-epoch',
                        default=1000,
                        type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p',
                        '--print-freq',
                        default=100,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument(
        '--per-class-eval',
        action='store_true',
        help='whether output per-class accuracy during evaluation')
    parser.add_argument(
        "--log",
        type=str,
        default='dann',
        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase",
                        type=str,
                        default='train',
                        choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                        "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
