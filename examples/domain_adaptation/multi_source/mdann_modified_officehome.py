from common.utils.analysis import collect_feature_and_labels, tsne, a_distance
from common.utils.logger import CompleteLogger
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.data import ForeverDataIterator
from common.vision.transforms import ResizeImage
import common.vision.models as models
from common.vision.datasets.modified_officehome import ModifiedOfficeHome
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


    print("Source(s): {} Target(s): {}".format(args.sources, args.targets))
    # assuming that args.sources and arg.targets are disjoint sets
    num_domains = len(args.sources) + len(args.targets)
    
    train_source_dataset = ModifiedOfficeHome(root=args.root,
                                              tasks=args.sources,
                                              download=True,
                                              transform=train_transform)
    # train_source_dataset = ConcatDataset([
    #     dataset(root=args.root,
    #             task=source,
    #             download=True,
    #             transform=train_transform) for source in args.sources
    # ])
    train_source_loader = DataLoader(train_source_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.workers,
                                     drop_last=True)
    train_target_dataset = ModifiedOfficeHome(root=args.root,
                                              tasks=args.targets,
                                              download=True,
                                              transform=train_transform)
    # train_target_dataset = ConcatDataset([
    #     dataset(root=args.root,
    #             task=target,
    #             download=True,
    #             transform=train_transform) for target in args.targets
    # ])
    train_target_loader = DataLoader(train_target_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.workers,
                                     drop_last=True)
    val_dataset = ModifiedOfficeHome(root=args.root,
                                     tasks=args.targets,
                                     download=True,
                                     transform=val_transform)
    # val_dataset = ConcatDataset([
    #     dataset(root=args.root,
    #             task=target,
    #             download=True,
    #             transform=val_transform) for target in args.targets
    # ])
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers)
    # if args.data == 'DomainNet':
    #     test_dataset = ModifiedOfficeHome(root=args.root,
    #                                       task=args.targets,
    #                                       split='test',
    #                                       download=True,
    #                                       transform=val_transform)
    #     test_loader = DataLoader(test_dataset,
    #                              batch_size=args.batch_size,
    #                              shuffle=False,
    #                              num_workers=args.workers)
    # else:
    test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(backbone,
                                 train_source_dataset.num_classes,
                                 bottleneck_dim=args.bottleneck_dim).to(device)
    # TODO: Make the domain discriminator work on multiple domains
    multidomain_discri = MultidomainDiscriminator(in_feature=classifier.features_dim,
                                                  hidden_size=1024, num_domains=num_domains).to(device)

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
        source_feature, source_labels = collect_feature_and_labels(train_source_loader,
                                         feature_extractor, device)
        target_feature, target_labels = collect_feature_and_labels(train_target_loader,
                                         feature_extractor, device)
        source_domain_labels = ModifiedOfficeHome.get_category(
            source_labels, classifier.num_classes)
        target_domain_labels = ModifiedOfficeHome.get_category(
            target_labels, classifier.num_classes)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, 
                        filename=tSNE_filename,
                        source_domain_labels=source_domain_labels, 
                        target_domain_labels=target_domain_labels,
                        num_domains=num_domains)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        # A_distance = a_distance.calculate(source_feature, target_feature,
        #                                   device)
        # print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args, 0)
        print(acc1)
        return

    # name of tensorboard
    comment = f' arch=resnet18 epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} lr_gamma={args.lr_gamma} lr_decay={args.lr_decay} trade_off={args.trade_off} seed={args.seed} domain={args.sources}2{args.targets}'
    # Summary writer for tensorboard
    tb = SummaryWriter(comment=comment)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, multidomain_adv,
              optimizer, lr_scheduler, epoch, args, tb)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args, epoch, tb)

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
    acc1 = validate(test_loader, classifier, args, epoch)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()
    tb.close()


def train(train_source_iter: ForeverDataIterator,
          train_target_iter: ForeverDataIterator, model: ImageClassifier,
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
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        # retrieve the class and domain from the modified office_home dataset
        class_labels_s = ModifiedOfficeHome.get_category(
            labels_s, model.num_classes)
        domain_labels_s = ModifiedOfficeHome.get_style(
            labels_s, model.num_classes)
        domain_labels_t = ModifiedOfficeHome.get_style(
            labels_t, model.num_classes)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        # labels_s = labels_s.to(device)
        # add new labels to device
        class_labels_s = class_labels_s.to(device)
        domain_labels_s = domain_labels_s.to(device)
        domain_labels_t = domain_labels_t.to(device)
        d_labels = torch.cat((domain_labels_s, domain_labels_t), dim=0)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        # f_s, f_t = f.chunk(2, dim=0)

        # Updating the loss functions with new labels
        # cls_loss = F.cross_entropy(y_s, labels_s)
        cls_loss = F.cross_entropy(y_s, class_labels_s)
        # TODO: Make a new domain discriminator for multiple class labels

        transfer_loss = multidomain_adv(f, d_labels)
        # transfer_loss = multidomain_adv(f_s, f_t, domain_labels_s, domain_labels_t)
        domain_acc = multidomain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, class_labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))
        # seperated loss between two heads
        cls_losses.update(cls_loss.item(), x_s.size(0))
        transfer_losses.update(transfer_loss.item(), x_s.size(0))

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
    tb.add_scalar('Classification Loss', cls_losses.sum, epoch)
    tb.add_scalar('Domain Disriminator Loss', transfer_losses.sum, epoch)
    tb.add_scalar('Total Loss', losses.sum, epoch)
    tb.add_scalar('Classification Accuracy', cls_accs.avg, epoch)
    tb.add_scalar('Domain Discrimination Accuracy', domain_accs.avg, epoch)
    # tb.add_histogram('Classification Softmax', TODO, epoch)


def validate(val_loader: DataLoader,
             model: ImageClassifier,
             args: argparse.Namespace,
             epoch: int,
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
            target = ModifiedOfficeHome.get_category(target, model.num_classes)
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
        tb.add_scalar('Validation Classification Loss', losses.sum, epoch)
        tb.add_scalar('Validation Classification Accuracy', top1.avg, epoch)

    return top1.avg


if __name__ == '__main__':
    architecture_names = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(
        description='MDANN for Mulisource and Multidomain Adpation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument('-s', '--sources', nargs='+', help='source domain(s)')
    parser.add_argument('-t', '--targets', nargs='+', help='target domain(s)')
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
