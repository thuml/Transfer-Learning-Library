import random
import time
import warnings
import sys
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

sys.path.append('../../..')
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import MultipleApply
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.data import ForeverDataIterator
from common.utils.logger import CompleteLogger
from ftlib.finetune.bi_tuning import Classifier, Bituning

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
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
    train_augmentation = [
        T.RandomResizedCrop(224, scale=(0.2, 1.)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ]
    train_transform = MultipleApply([T.Compose(train_augmentation), T.Compose(train_augmentation)])

    dataset = datasets.__dict__[args.data]
    train_dataset = dataset(root=args.root, split='train', sample_rate=args.sample_rate,
                            download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    train_iter = ForeverDataIterator(train_loader)
    val_dataset = dataset(root=args.root, split='test', sample_rate=100, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    num_classes = train_dataset.num_classes
    backbone_q = models.__dict__[args.arch](pretrained=True)
    if args.pretrained:
        print("=> loading pre-trained backbone from '{}'".format(args.pretrained))
        pretrained_dict = torch.load(args.pretrained)
        backbone_q.load_state_dict(pretrained_dict, strict=False)
    classifier_q = Classifier(backbone_q, num_classes, projection_dim=args.projection_dim)
    if args.pretrained_fc:
        print("=> loading pre-trained fc from '{}'".format(args.pretrained_fc))
        pretrained_fc_dict = torch.load(args.pretrained_fc)
        classifier_q.projector.load_state_dict(pretrained_fc_dict, strict=False)
    classifier_q = classifier_q.to(device)
    backbone_k = models.__dict__[args.arch](pretrained=True)
    classifier_k = Classifier(backbone_k, num_classes).to(device)

    bituning = Bituning(classifier_q, classifier_k, num_classes, K=args.K, m=args.m, T=args.T)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier_q.get_parameters(args.lr), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, gamma=args.lr_gamma)

    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier_q.load_state_dict(checkpoint)
        acc1 = validate(val_loader, classifier_q, args)
        print(acc1)
        return

    # start training
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_iter, bituning, optimizer, epoch, args)
        lr_scheduler.step()

        # evaluate on validation set
        acc1 = validate(val_loader, classifier_q, args)

        # remember best acc@1 and save checkpoint
        torch.save(classifier_q.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    logger.close()


def train(train_iter: ForeverDataIterator, bituning, optimizer: SGD, epoch:int, args:argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    contrastive_losses = AverageMeter('Contrastive Loss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, contrastive_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)
    contrastive_criterion = torch.nn.KLDivLoss(reduction='batchmean').to(device)

    # switch to train mode
    bituning.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x, labels = next(train_iter)
        img_q, img_k = x[0], x[1]

        img_q = img_q.to(device)
        img_k = img_k.to(device)
        labels = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, logits_z, logits_y, bituning_labels = bituning(img_q, img_k, labels)
        cls_loss = classifier_criterion(y, labels)
        contrastive_loss_z = contrastive_criterion(logits_z, bituning_labels)
        contrastive_loss_y = contrastive_criterion(logits_y, bituning_labels)
        contrastive_loss = (contrastive_loss_z + contrastive_loss_y)
        loss = cls_loss + contrastive_loss * args.trade_off

        # measure accuracy and record loss
        losses.update(loss.item(), x[0].size(0))
        cls_losses.update(cls_loss.item(), x[0].size(0))
        contrastive_losses.update(contrastive_loss.item(), x[0].size(0))

        cls_acc = accuracy(y, labels)[0]
        cls_accs.update(cls_acc.item(), x[0].size(0))

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
            output = model(images)
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
    parser = argparse.ArgumentParser(description='Bi-tuning for Finetuning')
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
    parser.add_argument('--pretrained', default=None,
                        help="pretrained checkpoint of the backbone. "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument('--pretrained-fc', default=None,
                        help="pretrained checkpoint of the fc. "
                             "(default: None)")
    parser.add_argument('--T', default=0.07, type=float, help="temperature. (default: 0.07)")
    parser.add_argument('--K', type=int, default=40, help="queue size. (default: 40)")
    parser.add_argument('--m', type=float, default=0.999, help="momentum coefficient. (default: 0.999)")
    parser.add_argument('--projection-dim', type=int, default=128,
                        help="dimension of the projection head. (default: 128)")
    parser.add_argument('--trade-off', type=float, default=1.0, help="trade-off parameters. (default: 1.0)")

    # training parameters
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N',
                        help='mini-batch size (default: 48)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay-epochs', type=int, default=(12,), nargs='+', help='epochs to decay lr')
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
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    args = parser.parse_args()
    main(args)
