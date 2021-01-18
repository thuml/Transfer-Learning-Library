import random
import time
import warnings
import sys
import argparse
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.grl import WarmStartGradientReverseLayer
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation._util import entropy
from dalib.adaptation.importance_weighted_adversarial_nets import ImageClassifier, ImageClassifierHead
import dalib.vision.datasets.partial as datasets
from dalib.vision.datasets.partial import default_partial as partial
import dalib.vision.models as models
from dalib.utils.data import ForeverDataIterator
from dalib.utils.metric import accuracy, ConfusionMatrix
from dalib.utils.avgmeter import AverageMeter, ProgressMeter
from dalib.vision.transforms import ResizeImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.center_crop:
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    val_transform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    partial_dataset = partial(dataset)
    train_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = partial_dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = partial_dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = partial_dataset(root=args.root, task=args.target, split='test', download=True,
                                       transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    num_classes = train_source_dataset.num_classes

    backbone_s = models.__dict__[args.arch](pretrained=True)
    backbone_t = models.__dict__[args.arch](pretrained=True)

    # use separate classifiers for source(classifier_s) and target(classifier_t) domain
    classifier_head = ImageClassifierHead(args.bottleneck_dim, num_classes).to(device)
    # According to 'Importance Weighted Adversarial Nets for Partial Domain Adaptation'
    # feature extractor F_s, F_t must be different but classifier C is shared(**head** here)
    classifier_s = ImageClassifier(backbone_s, num_classes, args.bottleneck_dim, head=classifier_head).to(device)
    classifier_t = ImageClassifier(backbone_t, num_classes, args.bottleneck_dim, head=classifier_head).to(device)

    # stage 1: pretrain feature_extractor F_s and classifier C
    pretrain_classifier(classifier_s, train_source_dataset, args)

    # initialize F_t using the parameters of F_s
    classifier_t.load_state_dict(classifier_s.state_dict())
    D = DomainDiscriminator(in_feature=classifier_s.features_dim, hidden_size=1024).to(device)
    D_0 = DomainDiscriminator(in_feature=classifier_s.features_dim, hidden_size=1024).to(device)

    # stage 2: train feature_extractor F_t and domain classifier D, D_0 simultaneously
    train(train_source_iter, train_target_iter, classifier_s, classifier_t, D, D_0, val_loader, test_loader, args)


def pretrain_classifier(model: ImageClassifier, dataset, args: argparse.Namespace):
    """In this function we perform supervised pretraining
    Inputs:
        **model**: model to pretrain(classifier_s)
        **dataset**: data from source domain
    """
    num_samples = len(dataset)
    num_samples_train = num_samples * 4 // 5
    # split dataset into training_set and validation_set
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [num_samples_train, num_samples - num_samples_train])

    train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=36, shuffle=False, num_workers=4)

    train_iter = ForeverDataIterator(train_loader)

    # parameters for pretraining
    optimizer = SGD(model.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # start training
    print('pretrain feature extractor F_s and classifier C on source domain.')
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # statistics
        losses = AverageMeter('Loss', ':6.2f')
        cls_accs = AverageMeter('Cls Acc', ':3.1f')
        progress = ProgressMeter(
            args.iters_per_epoch,
            [losses, cls_accs],
            prefix="Epoch: [{}]".format(epoch))
        # switch to train mode
        model.train()

        for i in range(args.iters_per_epoch):
            x, labels = next(train_iter)
            x = x.to(device)
            labels = labels.to(device)
            
            y, _ = model(x)

            loss = F.cross_entropy(y, labels)
            cls_acc = accuracy(y, labels)[0]
            losses.update(loss.item(), x.size(0))
            cls_accs.update(cls_acc.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if i % args.print_freq == 0:
                progress.display(i)

        # evaluate on validation set
        acc1 = validate(val_loader, model, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(model.state_dict())
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    model.load_state_dict(best_model)


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, classifier_s: ImageClassifier,
          classifier_t: ImageClassifier, D: DomainDiscriminator, D_0: DomainDiscriminator, val_loader: DataLoader,
          test_loader: DataLoader, args: argparse.Namespace):
    # In this function we train feature_extractor F_t and domain classifier D, D_0 simultaneously

    # only train feature extractors(backbone, bottleneck) of classifier_t
    classifier_s.eval()
    classifier_t.train()
    classifier_t.head.eval()
    classifier_t.set_features_only(True)

    domain_adv_D = DomainAdversarialLoss(D).to(device)
    # define trade off scheduler
    grl = WarmStartGradientReverseLayer(alpha=1, lo=0, hi=args.trade_off, max_iters=args.epochs * args.iters_per_epoch)
    domain_adv_D_0 = DomainAdversarialLoss(D_0, grl=grl).to(device)
    # parameters
    optimizer_D = SGD(D.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer_D_0 = SGD(classifier_t.get_parameters() + D_0.get_parameters(), args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler_D = LambdaLR(optimizer_D, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_D_0 = LambdaLR(optimizer_D_0, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # start training
    print("train feature extractor F_t and domain classifier D, D_0 simultaneously.")
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # statistics
        batch_time = AverageMeter('Time', ':5.2f')
        data_time = AverageMeter('Data', ':5.2f')
        losses = AverageMeter('Loss', ':6.2f')
        cls_accs = AverageMeter('Cls Acc', ':3.1f')
        domain_accs_D = AverageMeter('Domain Acc for D', ':3.1f')
        domain_accs_D_0 = AverageMeter('Domain Acc for D_0', ':3.1f')
        tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

        progress = ProgressMeter(
            args.iters_per_epoch,
            [batch_time, data_time, losses, cls_accs, domain_accs_D, domain_accs_D_0, tgt_accs],
            prefix="Epoch: [{}]".format(epoch))

        # swtich to train mode
        # notice classifier_t.head should be set to eval mode
        classifier_t.train()
        classifier_t.head.eval()
        domain_adv_D.train()
        domain_adv_D_0.train()

        end = time.time()
        for i in range(args.iters_per_epoch):
            x_s, labels_s = next(train_source_iter)
            x_t, labels_t = next(train_target_iter)

            x_s = x_s.to(device)
            x_t = x_t.to(device)
            labels_s = labels_s.to(device)
            labels_t = labels_t.to(device)
            
            # measure data loading time
            data_time.update(time.time() - end)
            
            y_s, f_s = classifier_s(x_s)
            y_t, f_t = classifier_t(x_t)

            # 1. Optimize D,
            D.train()
            domain_cls_loss = domain_adv_D(f_s.detach(), f_t.detach())
            # compute gradient and do SGD step
            optimizer_D.zero_grad()
            domain_cls_loss.backward()
            optimizer_D.step()
            lr_scheduler_D.step()

            # 2. Get importance weights
            D.eval()
            weights = 1. - D(f_s).detach()
            weights = weights / weights.mean()
            
            # 3. Optimize F_t, D_0,
            y_t = F.softmax(y_t, dim=1)
            entropy_loss = args.gamma * torch.mean(entropy(y_t))
            transfer_loss = domain_adv_D_0(f_s, f_t, weights)
            loss = entropy_loss + transfer_loss
            # compute gradient and do SGD step
            optimizer_D_0.zero_grad()
            classifier_t.head.zero_grad()
            loss.backward()
            optimizer_D_0.step()
            lr_scheduler_D_0.step()

            cls_acc = accuracy(y_s, labels_s)[0]
            tgt_acc = accuracy(y_t, labels_t)[0]
            
            losses.update(loss.item(), x_s.size(0))
            cls_accs.update(cls_acc.item(), x_s.size(0))
            tgt_accs.update(tgt_acc.item(), x_s.size(0))
            domain_accs_D.update(domain_adv_D.domain_discriminator_accuracy, x_s.size(0))
            domain_accs_D_0.update(domain_adv_D_0.domain_discriminator_accuracy, x_s.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier_t, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier_t.state_dict())
        best_acc1 = max(acc1, best_acc1)

    classifier_t.load_state_dict(best_model)
    acc1 = validate(test_loader, classifier_t, args)
    print("test_acc1 = {:3.1f}".format(acc1))


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace):
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

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of source (and target) dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # all parameters below correspond to the second state of training, the adversarial training process
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='the trade-off hyper-parameter for entropy loss(default: 0.01)')
    parser.add_argument('--trade-off', default=0.1, type=float,
                        help='the trade-off hyper-parameter for transfer loss(default: 0.1))')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--center-crop', default=False, action='store_true')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    args = parser.parse_args()
    print(args)
    main(args)
