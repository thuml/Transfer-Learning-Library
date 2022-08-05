"""
@author: Jinghan Gao, Baixu Chen
@contact: getterk@163.com, cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T

import utils
import tllib.vision.datasets.universal as datasets
from tllib.vision.datasets.universal import default_universal as universal
from tllib.alignment.cmu import ImageClassifier, Ensemble, norm, cal_f1, get_marginal_confidence, get_entropy
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ens_transforms = [
    T.Compose([T.Resize(256),
               T.RandomHorizontalFlip(),
               T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2,
                              interpolation=T.InterpolationMode.BICUBIC, fill=(255, 255, 255)),
               T.CenterCrop(224),
               T.RandomGrayscale(p=0.5),
               T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(256),
               T.RandomHorizontalFlip(),
               T.RandomPerspective(),
               T.FiveCrop(224),
               T.Lambda(lambda crops: crops[0]),
               T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
               T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(256),
               T.RandomHorizontalFlip(),
               T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2,
                              interpolation=T.InterpolationMode.BICUBIC, fill=(255, 255, 255)),
               T.FiveCrop(224),
               T.Lambda(lambda crops: crops[1]),
               T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
               T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(256),
               T.RandomHorizontalFlip(),
               T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1,
                              interpolation=T.InterpolationMode.BICUBIC, fill=(255, 255, 255)),
               T.RandomPerspective(),
               T.FiveCrop(224),
               T.Lambda(lambda crops: crops[2]),
               T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(256),
               T.RandomHorizontalFlip(),
               T.RandomPerspective(),
               T.FiveCrop(224),
               T.Lambda(lambda crops: crops[3]),
               T.RandomGrayscale(p=0.5),
               T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
]


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
    train_transform = utils.get_train_transform(args.train_resizing)
    val_transform = utils.get_val_transform(args.val_resizing)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    dataset = datasets.__dict__[args.data]
    source_dataset = universal(dataset, source=True)
    target_dataset = universal(dataset, source=False)
    train_source_dataset = source_dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_target_dataset = target_dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    ens_datasets = [source_dataset(root=args.root, task=args.source, download=True, transform=ens_transforms[i]) for i
                    in range(5)]
    val_dataset = target_dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    if args.data == 'DomainNet':
        test_dataset = target_dataset(root=args.root, task=args.target, split='test', download=True,
                                      transform=val_transform)
    else:
        test_dataset = val_dataset
    source_classes = np.unique(train_source_dataset.targets)
    num_classes = len(source_classes)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    ens_iters = [ForeverDataIterator(
        DataLoader(ens_datasets[i], batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True))
        for i in range(5)]

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer).to(
        device)
    ens_classifier = Ensemble(classifier.features_dim, train_source_dataset.num_classes).to(device)

    if not os.path.exists("{}/stage1_models".format(args.log)):
        os.mkdir("{}/stage1_models".format(args.log))

    pretrain_model_path = "{}/stage1_models/pretrain.pth".format(args.log, args.source)
    if not os.path.exists(pretrain_model_path):
        # pretrain the classifier and ens_classifier
        optimizer_pretrain = SGD(classifier.get_parameters() + ens_classifier.get_parameters(), args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        lr_lambda = lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
        lr_scheduler_pretrain = LambdaLR(optimizer_pretrain, lr_lambda)

        best_f1 = 0
        for epoch in range(args.epochs):
            pretrain(train_source_iter, ens_iters, classifier, ens_classifier, optimizer_pretrain, args, epoch,
                     lr_scheduler_pretrain)

            f1 = cal_f1(val_loader, classifier, ens_classifier, source_classes, device)

            if best_f1 < f1:
                state = {'classifier': classifier.state_dict(), 'ens_classifier': ens_classifier.state_dict()}
                torch.save(state, pretrain_model_path)
                best_f1 = f1

        print("Best F1 {:.4f}".format(best_f1))
        exit(0)

    # domain discriminator
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(), args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)
    lr_lambda = lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    ens_optimizer = SGD(ens_classifier.get_parameters(), args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay, nesterov=True)
    ens_lr_scheduler = [LambdaLR(ens_optimizer, lr_lambda)] * 5

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    target_score_upper = torch.zeros(1).to(device)
    target_score_lower = torch.zeros(1).to(device)

    # calculate source weight
    checkpoint = torch.load(pretrain_model_path)
    classifier.load_state_dict(checkpoint['classifier'])
    ens_classifier.load_state_dict(checkpoint['ens_classifier'])
    source_class_weight = evaluate_source_common(val_loader, classifier, ens_classifier, source_classes, args)

    mask = torch.where(source_class_weight > args.cut)
    source_class_weight = torch.zeros_like(source_class_weight)
    source_class_weight[mask] = 1
    print('Weight of each source class')
    print(source_class_weight)

    if args.phase != 'train':
        classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best_classifier')))
        ens_classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best_ens_classifier')))

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
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
        acc1, h_score = validate(test_loader, classifier, ens_classifier, source_classes, args)
        return

    # start training
    best_acc = 0.
    best_h_score = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        target_score_upper, target_score_lower = train(train_source_iter, train_target_iter, classifier, domain_adv,
                                                       ens_classifier, optimizer, lr_scheduler, epoch,
                                                       source_class_weight, target_score_upper, target_score_lower,
                                                       args)

        for i in range(5):
            train_ens_classifier(ens_iters[i], classifier, ens_classifier, ens_optimizer, ens_lr_scheduler[i], epoch,
                                 args, i)

        # evaluate on validation set
        acc, h_score = validate(val_loader, classifier, ens_classifier, source_classes, args)
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest_classifier'))
        torch.save(ens_classifier.state_dict(), logger.get_checkpoint_path('latest_ens_classifier'))

        best_acc = max(acc, best_acc)
        if h_score > best_h_score:
            best_h_score = h_score
            # remember best h_score and save checkpoint
            shutil.copy(logger.get_checkpoint_path('latest_classifier'), logger.get_checkpoint_path('best_classifier'))
            shutil.copy(logger.get_checkpoint_path('latest_ens_classifier'),
                        logger.get_checkpoint_path('best_ens_classifier'))

    print('* Val Best Mean Acc@1 {:.3f}'.format(best_acc))
    print('* Val Best H-score {:.3f}'.format(best_h_score))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best_classifier')))
    ens_classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best_ens_classifier')))
    test_acc, test_h_score = validate(test_loader, classifier, ens_classifier, source_classes, args)
    print('* Test Mean Acc@1 {:.3f} H-score {:.3f}'.format(test_acc, test_h_score))
    logger.close()


def pretrain(train_source_iter: ForeverDataIterator, ens_iters, model, ens_classifier, optimizer, args, epoch,
             lr_scheduler):
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, cls_accs],
        prefix="Pretrain Epoch: [{}]".format(epoch))

    model.train()
    ens_classifier.train()

    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)
        y_s, f_s = model(x_s)
        cls_loss = F.cross_entropy(y_s, labels_s)

        ens_losses = []
        for j, ens_iter in enumerate(ens_iters):
            x_se, labels_se = next(ens_iter)
            x_se = x_se.to(device)
            labels_se = labels_se.to(device)
            y_se, f_se = model(x_se)
            y_se = ens_classifier(f_se, index=j)
            ens_losses.append(F.cross_entropy(y_se, labels_se))

        cls_acc = accuracy(y_se, labels_se)[0]
        cls_accs.update(cls_acc.item(), args.batch_size)

        loss = sum(ens_losses) + cls_loss
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % args.print_freq == 0:
            progress.display(i)


def train_ens_classifier(train_source_iter, model, ens_classifier, optimizer, lr_scheduler, epoch, args,
                         classifier_index):
    losses = AverageMeter('Loss', ':4.2f')
    cls_accs = AverageMeter('Cls Acc', ':5.1f')
    progress = ProgressMeter(
        args.iters_per_epoch // 2,
        [losses, cls_accs],
        prefix="Train ensemble classifier {}:, Epoch: [{}]".format(classifier_index + 1, epoch))

    model.eval()
    ens_classifier.train()

    for i in range(args.iters_per_epoch // 2):
        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # compute output
        with torch.no_grad():
            y_s, f_s = model(x_s)
        y_s = ens_classifier(f_s, classifier_index)
        loss = F.cross_entropy(y_s, labels_s)
        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % args.print_freq == 0:
            progress.display(i)


def evaluate_source_common(val_loader, model, ens_classifier, source_classes, args):
    temperature = 1
    # switch to evaluate mode
    model.eval()
    ens_classifier.eval()

    common = []
    target_private = []
    all_confidence = []
    all_entropy = []
    all_labels = []
    all_output = []

    source_weight = torch.zeros(len(source_classes)).to(device)
    cnt = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            output, f = model(images)
            output = F.softmax(output, -1) / temperature

            yt_1, yt_2, yt_3, yt_4, yt_5 = ens_classifier(f, -1)
            confidence = get_marginal_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)

            all_confidence.extend(confidence)
            all_entropy.extend(entropy)
            all_labels.extend(labels)
            all_output.extend(output)

    all_confidence = norm(torch.tensor(all_confidence))
    all_entropy = norm(torch.tensor(all_entropy))
    all_score = (all_confidence + 1 - all_entropy) / 2

    print('source threshold {:.3f}'.format(args.src_threshold))

    for i in range(len(all_score)):
        if all_score[i] >= args.src_threshold:
            source_weight += all_output[i]
            cnt += 1
        if all_labels[i].item() in source_classes:
            common.append(all_score[i])
        else:
            target_private.append(all_score[i])

    source_weight = norm(source_weight / cnt)
    return source_weight


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          domain_adv: DomainAdversarialLoss, ens_classifier: Ensemble, optimizer: SGD, lr_scheduler: LambdaLR,
          epoch: int, source_class_weight, target_score_upper, target_score_lower, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    losses = AverageMeter('Loss', ':4.2f')
    cls_accs = AverageMeter('Cls Acc', ':4.1f')
    domain_accs = AverageMeter('Domain Acc', ':4.1f')
    score_upper = AverageMeter('Score Upper', ':4.2f')
    score_lower = AverageMeter('Score Lower', ':4.2f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, losses, cls_accs, domain_accs, score_upper, score_lower],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()
    ens_classifier.eval()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        with torch.no_grad():
            yt_1, yt_2, yt_3, yt_4, yt_5 = ens_classifier(f_t, -1)
            confidence = get_marginal_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)
            w_t = (confidence + 1 - entropy) / 2
            target_score_upper = target_score_upper * 0.01 + w_t.max() * 0.99
            target_score_lower = target_score_lower * 0.01 + w_t.min() * 0.99
            w_t = (w_t - target_score_lower) / (target_score_upper - target_score_lower)
            w_s = torch.tensor([source_class_weight[i] for i in labels_s]).to(device)

        loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t, w_s.detach(), w_t.detach())
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss += transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), args.batch_size)
        cls_accs.update(cls_acc.item(), args.batch_size)
        domain_accs.update(domain_acc.item(), args.batch_size)
        score_upper.update(target_score_upper.item(), 1)
        score_lower.update(target_score_lower.item(), 1)

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

    return target_score_upper, target_score_lower


def validate(val_loader, model, ens_classifier, source_classes, args):
    # switch to evaluate mode
    model.eval()
    ens_classifier.eval()

    all_confidence = []
    all_entropy = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            output, f = model(images)
            _, predictions = torch.max(F.softmax(output, -1), 1)

            yt_1, yt_2, yt_3, yt_4, yt_5 = ens_classifier(f, -1)
            confidence = get_marginal_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)

            all_confidence.extend(confidence)
            all_entropy.extend(entropy)
            all_predictions.extend(predictions)
            all_labels.extend(labels)

    all_confidence = norm(torch.tensor(all_confidence))
    all_entropy = norm(torch.tensor(all_entropy))
    all_score = (all_confidence + 1 - all_entropy) / 2

    counters = utils.AccuracyCounter(len(source_classes) + 1)
    for (predictions, labels, score) in zip(all_predictions, all_labels, all_score):
        labels = labels.item()
        if labels in source_classes:
            counters.add_total(labels)
            if score >= args.threshold and predictions == labels:
                counters.add_correct(labels)
        else:
            counters.add_total(-1)
            if score < args.threshold:
                counters.add_correct(-1)

    print('* Acc@1 of each class')
    print(counters.per_class_accuracy())
    print('* Mean Acc@1 {:.3f}'.format(counters.mean_accuracy()))
    print('* H-score {:.3f}'.format(counters.h_score()))

    return counters.mean_accuracy(), counters.h_score()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CMU for Universal Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain')
    parser.add_argument('-t', '--target', help='target domain')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--threshold', default=0.8, type=float,
                        help='When class confidence is less than the given threshold, '
                             'model will output "unknown" (default: 0.5)')
    parser.add_argument('--src-threshold', default=0.8, type=float,
                        help='threshold for source common class item counting')
    parser.add_argument('--cut', default=0.2, type=float,
                        help='cut threshold for common classes identifying')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cmu',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
