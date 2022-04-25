"""
@author: Jinghan Gao
@contact: getterk@163.com
"""
import argparse
import os
import os.path as osp
import random
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import utils
from examples.domain_adaptation.openset_domain_adaptation.utils import get_model, get_train_transform, \
    get_val_transform, get_dataset_names, get_model_names
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.utils.analysis import collect_feature, tsne, a_distance
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger

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
    train_transform = get_train_transform(args.train_resizing)
    val_transform = get_val_transform(args.val_resizing)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, esem_datasets, num_classes, source_classes = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = get_model(args.arch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = utils.ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer)\
        .to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    esem = utils.Ensemble(classifier.features_dim, train_source_dataset.num_classes).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_lambda = lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    optimizer_esem = SGD(esem.get_parameters(), args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay, nesterov=True)
    esem_lr_schedulers = [LambdaLR(optimizer_esem, lr_lambda)] * 5

    optimizer_pre = SGD(esem.get_parameters() + classifier.get_parameters(), args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler_pre = LambdaLR(optimizer_pre, lr_lambda)

    esem_iters = utils.get_esem_data_iters(esem_datasets, batch_size=args.batch_size, num_workers=args.workers)

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    if not os.path.exists(f"{args.log}/stage1_models"):
        os.mkdir(f"{args.log}/stage1_models")

    pretrain_model_path = f"{args.log}/stage1_models/scw_{args.source}_pretrain.pth"
    if not os.path.exists(pretrain_model_path):
        best_f1 = 0
        for epoch in range(args.epochs):
            utils.pretrain(train_source_iter, esem_iters, classifier, esem, optimizer_pre, args, epoch,
                           lr_scheduler_pre, device)

            f1 = utils.cal_f1(val_loader, classifier, esem, source_classes, device)
            print(f"Got Best F1 {f1:.4f}")

            if best_f1 < f1:
                state = {'classifier': classifier.state_dict(), 'esem': esem.state_dict()}
                torch.save(state, pretrain_model_path)
                best_f1 = f1

        exit(0)
    else:
        checkpoint = torch.load(pretrain_model_path)

        classifier.load_state_dict(checkpoint['classifier'])
        esem.load_state_dict(checkpoint['esem'])

        source_class_weight = utils.evaluate_source_common(val_loader, classifier, esem, source_classes, args, device)

    target_score_upper = torch.zeros(1).to(device)
    target_score_lower = torch.zeros(1).to(device)

    mask = torch.where(source_class_weight > args.cut)
    source_class_weight = torch.zeros_like(source_class_weight)
    source_class_weight[mask] = 1
    print(source_class_weight)

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
        acc1, _ = utils.validate(val_loader, classifier, esem, source_classes, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    best_acc2 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        target_score_upper, target_score_lower = utils.train(train_source_iter, train_target_iter, classifier,
                                                             domain_adv,
                                                             esem, optimizer, lr_scheduler, epoch, source_class_weight,
                                                             target_score_upper, target_score_lower, args, device)

        for i in range(5):
            utils.train_esem(esem_iters[i], classifier, esem, optimizer_esem, esem_lr_schedulers[i], epoch, args,
                             i, device)

        # evaluate on validation set
        acc1, acc2 = utils.validate(val_loader, classifier, esem, source_classes, args, device)

        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_acc1 = max(acc1, best_acc1)
            best_acc2 = max(acc2, best_acc2)
            # remember best acc@1 and save checkpoint
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1, acc2 = utils.validate(test_loader, classifier, esem, source_classes, args, device)
    print("best_acc = {:3.3f}   {:3.3f}".format(acc1, acc2))

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CMU for Universal Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=get_dataset_names(),
                        help='dataset: ' + ' | '.join(get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain')
    parser.add_argument('-t', '--target', help='target domain')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(get_model_names()) +
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
