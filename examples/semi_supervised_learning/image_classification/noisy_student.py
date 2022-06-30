"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import copy
import random
import time
import warnings
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import utils
from tllib.vision.models.reid.loss import CrossEntropyLoss
from tllib.modules.classifier import Classifier
from tllib.vision.transforms import MultipleApply
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        self.dropout = nn.Dropout(0.5)
        self.as_teacher_model = False

    def forward(self, x: torch.Tensor):
        """"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        if not self.as_teacher_model:
            f = self.dropout(f)
        predictions = self.head(f)
        return predictions


def calc_teacher_output(classifier_teacher: ImageClassifier, weak_augmented_unlabeled_dataset):
    """Compute outputs of the teacher network. Here, we use weak data augmentation and do not introduce an additional
    dropout layer according to the Noisy Student paper `Self-Training With Noisy Student Improves ImageNet
    Classification <https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves
    _ImageNet_Classification_CVPR_2020_paper.pdf>`_.
    """

    data_loader = DataLoader(weak_augmented_unlabeled_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False)
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time],
        prefix='Computing teacher output: ')

    teacher_output = []
    with torch.no_grad():
        end = time.time()
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            output = classifier_teacher(images)
            teacher_output.append(output)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    teacher_output = torch.cat(teacher_output, dim=0)
    return teacher_output


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
    weak_augment = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    strong_augment = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                               auto_augment=args.auto_augment,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std)
    labeled_train_transform = MultipleApply([weak_augment, strong_augment])
    val_transform = utils.get_val_transform(args.val_resizing, norm_mean=args.norm_mean, norm_std=args.norm_std)
    print('labeled_train_transform: ', labeled_train_transform)
    print('weak_augment (input transform for teacher model): ', weak_augment)
    print('strong_augment (input transform for student model): ', strong_augment)
    print('val_transform:', val_transform)

    labeled_train_dataset, weak_augmented_unlabeled_dataset, val_dataset = \
        utils.get_dataset(args.data,
                          args.num_samples_per_class,
                          args.root, labeled_train_transform,
                          val_transform,
                          unlabeled_train_transform=weak_augment,
                          seed=args.seed)
    _, strong_augmented_unlabeled_dataset, _ = \
        utils.get_dataset(args.data,
                          args.num_samples_per_class,
                          args.root, labeled_train_transform,
                          val_transform,
                          unlabeled_train_transform=strong_augment,
                          seed=args.seed)

    strong_augmented_unlabeled_dataset = utils.convert_dataset(strong_augmented_unlabeled_dataset)
    print("labeled_dataset_size: ", len(labeled_train_dataset))
    print('unlabeled_dataset_size: ', len(weak_augmented_unlabeled_dataset))
    print("val_dataset_size: ", len(val_dataset))

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True)
    unlabeled_train_loader = DataLoader(strong_augmented_unlabeled_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)
    unlabeled_train_iter = ForeverDataIterator(unlabeled_train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone)
    num_classes = labeled_train_dataset.num_classes
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer,
                                 finetune=args.finetune).to(device)
    print(classifier)

    if args.pretrained_teacher:
        # load teacher model
        classifier_teacher = copy.deepcopy(classifier)
        checkpoint = torch.load(args.pretrained_teacher)
        classifier_teacher.load_state_dict(checkpoint)
        classifier_teacher.eval()
        classifier_teacher.as_teacher_model = True

        print('compute outputs of the teacher network')
        teacher_output = calc_teacher_output(classifier_teacher, weak_augmented_unlabeled_dataset)

    # define optimizer and lr scheduler
    if args.lr_scheduler == 'exp':
        optimizer = SGD(classifier.get_parameters(), args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        optimizer = SGD(classifier.get_parameters(base_lr=args.lr), args.lr, momentum=0.9, weight_decay=args.wd,
                        nesterov=True)
        lr_scheduler = utils.get_cosine_scheduler_with_warmup(optimizer, args.epochs * args.iters_per_epoch)

    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes)
        print(acc1)
        return

    # start training
    best_acc1 = 0.0
    best_avg = 0.0
    for epoch in range(args.epochs):
        # print lr
        print(lr_scheduler.get_lr())

        # train for one epoch
        if args.pretrained_teacher:
            train(labeled_train_iter, unlabeled_train_iter, classifier, teacher_output, optimizer, lr_scheduler,
                  epoch, args)
        else:
            utils.empirical_risk_minimization(labeled_train_iter, classifier, optimizer, lr_scheduler, epoch, args,
                                              device)

        # evaluate on validation set
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        best_avg = max(avg, best_avg)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    print('best_avg = {:3.1f}'.format(best_avg))
    logger.close()


def train(labeled_train_iter: ForeverDataIterator, unlabeled_train_iter: ForeverDataIterator, model, teacher_output,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    self_training_losses = AverageMeter('Self Training Loss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, self_training_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    self_training_criterion = CrossEntropyLoss().to(device)
    # switch to train mode
    model.train()

    end = time.time()
    batch_size = args.batch_size
    for i in range(args.iters_per_epoch):
        (x_l, x_l_strong), labels_l = next(labeled_train_iter)
        x_l = x_l.to(device)
        x_l_strong = x_l_strong.to(device)
        labels_l = labels_l.to(device)

        idx_u, (x_u_strong, _) = next(unlabeled_train_iter)
        idx_u = idx_u.to(device)
        x_u_strong = x_u_strong.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # clear grad
        optimizer.zero_grad()

        # compute output
        y_l = model(x_l)
        y_l_strong = model(x_l_strong)
        # cross entropy loss
        cls_loss = F.cross_entropy(y_l, labels_l) + args.trade_off_cls_strong * F.cross_entropy(y_l_strong, labels_l)
        cls_loss.backward()

        # self training loss
        y_u = teacher_output[idx_u]
        y_u_strong = model(x_u_strong)
        self_training_loss = args.trade_off_self_training * self_training_criterion(y_u_strong / args.T, y_u / args.T)
        self_training_loss.backward()

        # measure accuracy and record loss
        loss = cls_loss + self_training_loss
        losses.update(loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        self_training_losses.update(self_training_loss.item(), batch_size)

        cls_acc = accuracy(y_l, labels_l)[0]
        cls_accs.update(cls_acc.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noisy Student for Semi Supervised Learning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()))
    parser.add_argument('--num-samples-per-class', default=4, type=int,
                        help='number of labeled samples per class')
    parser.add_argument('--train-resizing', default='default', type=str)
    parser.add_argument('--val-resizing', default='default', type=str)
    parser.add_argument('--norm-mean', default=(0.485, 0.456, 0.406), type=float, nargs='+',
                        help='normalization mean')
    parser.add_argument('--norm-std', default=(0.229, 0.224, 0.225), type=float, nargs='+',
                        help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=utils.get_model_names(),
                        help='backbone architecture: ' + ' | '.join(utils.get_model_names()) + ' (default: resnet50)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int,
                        help='dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true', default=False,
                        help='no pool layer after the feature extractor')
    parser.add_argument('--pretrained-backbone', default=None, type=str,
                        help="pretrained checkpoint of the backbone "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='whether to use 10x smaller lr for backbone')
    parser.add_argument('--pretrained-teacher', default=None, type=str,
                        help='pretrained checkpoint of the teacher model')
    # training parameters
    parser.add_argument('--trade-off-cls-strong', default=0.1, type=float,
                        help='the trade-off hyper-parameter of cls loss on strong augmented labeled data')
    parser.add_argument('--trade-off-self-training', default=1, type=float,
                        help='the trade-off hyper-parameter of self training loss')
    parser.add_argument('--T', default=2, type=float,
                        help='temperature')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, metavar='LR', dest='lr',
                        help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='exp', type=str, choices=['exp', 'cos'],
                        help='learning rate decay strategy')
    parser.add_argument('--lr-gamma', default=0.0004, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default:5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run (default: 40)')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='number of iterations per epoch (default: 500)')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training ')
    parser.add_argument("--log", default='noisy_student', type=str,
                        help="where to save logs, checkpoints and debugging images")
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")
    args = parser.parse_args()
    main(args)
