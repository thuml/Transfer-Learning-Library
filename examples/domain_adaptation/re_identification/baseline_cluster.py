"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, DBSCAN

import utils
import tllib.vision.datasets.reid as datasets
from tllib.vision.datasets.reid.convert import convert_to_pytorch_dataset
from tllib.vision.models.reid.identifier import ReIdentifier
from tllib.vision.models.reid.loss import CrossEntropyLossWithLabelSmooth, SoftTripletLoss
from tllib.utils.metric.reid import extract_reid_feature, validate, visualize_ranked_results
from tllib.utils.data import ForeverDataIterator, RandomMultipleGallerySampler
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.height, args.width, args.train_resizing,
                                                random_horizontal_flip=True, random_color_jitter=False,
                                                random_gray_scale=False, random_erasing=True)
    val_transform = utils.get_val_transform(args.height, args.width)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    working_dir = osp.dirname(osp.abspath(__file__))
    source_root = osp.join(working_dir, args.source_root)
    target_root = osp.join(working_dir, args.target_root)

    # source dataset
    source_dataset = datasets.__dict__[args.source](root=osp.join(source_root, args.source.lower()))
    val_loader = DataLoader(
        convert_to_pytorch_dataset(list(set(source_dataset.query) | set(source_dataset.gallery)),
                                   root=source_dataset.images_dir,
                                   transform=val_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    # target dataset
    target_dataset = datasets.__dict__[args.target](root=osp.join(target_root, args.target.lower()))
    cluster_loader = DataLoader(
        convert_to_pytorch_dataset(target_dataset.train, root=target_dataset.images_dir, transform=val_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        convert_to_pytorch_dataset(list(set(target_dataset.query) | set(target_dataset.gallery)),
                                   root=target_dataset.images_dir,
                                   transform=val_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    # create model
    num_classes = args.num_clusters
    backbone = utils.get_model(args.arch)
    pool_layer = nn.Identity() if args.no_pool else None
    model = ReIdentifier(backbone, num_classes, finetune=args.finetune, pool_layer=pool_layer).to(device)
    model = DataParallel(model)

    # load pretrained weights
    pretrained_model = torch.load(args.pretrained_model_path)
    utils.copy_state_dict(model, pretrained_model)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        utils.copy_state_dict(model, checkpoint['model'])

    # analysis the model
    if args.phase == 'analysis':
        # plot t-SNE
        utils.visualize_tsne(source_loader=val_loader, target_loader=test_loader, model=model,
                             filename=osp.join(logger.visualize_directory, 'analysis', 'TSNE.pdf'), device=device)
        # visualize ranked results
        visualize_ranked_results(test_loader, model, target_dataset.query, target_dataset.gallery, device,
                                 visualize_dir=logger.visualize_directory, width=args.width, height=args.height,
                                 rerank=args.rerank)
        return

    if args.phase == 'test':
        print("Test on Source domain:")
        validate(val_loader, model, source_dataset.query, source_dataset.gallery, device, cmc_flag=True,
                 rerank=args.rerank)
        print("Test on target domain:")
        validate(test_loader, model, target_dataset.query, target_dataset.gallery, device, cmc_flag=True,
                 rerank=args.rerank)
        return

    # define loss function
    criterion_ce = CrossEntropyLossWithLabelSmooth(num_classes).to(device)
    criterion_triplet = SoftTripletLoss(margin=args.margin).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        utils.copy_state_dict(model, checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1

    # start training
    best_test_mAP = 0.
    for epoch in range(args.start_epoch, args.epochs):
        # run clustering algorithm and generate pseudo labels
        if args.clustering_algorithm == 'kmeans':
            train_target_iter = run_kmeans(cluster_loader, model, target_dataset, train_transform, args)
        elif args.clustering_algorithm == 'dbscan':
            train_target_iter, num_classes = run_dbscan(cluster_loader, model, target_dataset, train_transform, args)

        # define cross entropy loss with current number of classes
        criterion_ce = CrossEntropyLossWithLabelSmooth(num_classes).to(device)

        # define optimizer
        optimizer = Adam(model.module.get_parameters(base_lr=args.lr, rate=args.rate), args.lr,
                         weight_decay=args.weight_decay)

        # train for one epoch
        train(train_target_iter, model, optimizer, criterion_ce, criterion_triplet, epoch, args)

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            # remember best mAP and save checkpoint
            torch.save(
                {
                    'model': model.state_dict(),
                    'epoch': epoch
                }, logger.get_checkpoint_path(epoch)
            )
            print("Test on target domain...")
            _, test_mAP = validate(test_loader, model, target_dataset.query, target_dataset.gallery, device,
                                   cmc_flag=True, rerank=args.rerank)
            if test_mAP > best_test_mAP:
                shutil.copy(logger.get_checkpoint_path(epoch), logger.get_checkpoint_path('best'))
            best_test_mAP = max(test_mAP, best_test_mAP)

    print("best mAP on target = {}".format(best_test_mAP))
    logger.close()


def run_kmeans(cluster_loader: DataLoader, model: DataParallel, target_dataset, train_transform,
               args: argparse.Namespace):
    # run kmeans clustering algorithm
    print('Clustering into {} classes'.format(args.num_clusters))
    feature_dict = extract_reid_feature(cluster_loader, model, device, normalize=True)
    feature = torch.stack(list(feature_dict.values())).cpu().numpy()
    km = KMeans(n_clusters=args.num_clusters, random_state=args.seed).fit(feature)
    cluster_labels = km.labels_
    cluster_centers = km.cluster_centers_
    print('Clustering finished')

    # normalize cluster centers and convert to pytorch tensor
    cluster_centers = torch.from_numpy(cluster_centers).float().to(device)
    cluster_centers = F.normalize(cluster_centers, dim=1)
    # reinitialize classifier head
    model.module.head.weight.data.copy_(cluster_centers)

    # generate training set with pseudo labels
    target_train_set = []
    for (fname, _, cid), label in zip(target_dataset.train, cluster_labels):
        target_train_set.append((fname, int(label), cid))

    sampler = RandomMultipleGallerySampler(target_train_set, args.num_instances)
    train_target_loader = DataLoader(
        convert_to_pytorch_dataset(target_train_set, root=target_dataset.images_dir, transform=train_transform),
        batch_size=args.batch_size, num_workers=args.workers, sampler=sampler, pin_memory=True, drop_last=True)
    train_target_iter = ForeverDataIterator(train_target_loader)

    return train_target_iter


def run_dbscan(cluster_loader: DataLoader, model: DataParallel, target_dataset, train_transform,
               args: argparse.Namespace):
    # run dbscan clustering algorithm
    feature_dict = extract_reid_feature(cluster_loader, model, device, normalize=True)
    feature = torch.stack(list(feature_dict.values())).cpu()
    rerank_dist = utils.compute_rerank_dist(feature).numpy()

    print('Clustering with dbscan algorithm')
    dbscan = DBSCAN(eps=0.6, min_samples=4, metric='precomputed', n_jobs=-1)
    cluster_labels = dbscan.fit_predict(rerank_dist)
    print('Clustering finished')

    # generate training set with pseudo labels and calculate cluster centers
    target_train_set = []
    cluster_centers = {}
    for i, ((fname, _, cid), label) in enumerate(zip(target_dataset.train, cluster_labels)):
        if label == -1:
            continue
        target_train_set.append((fname, label, cid))

        if label not in cluster_centers:
            cluster_centers[label] = []
        cluster_centers[label].append(feature[i])

    cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
    cluster_centers = torch.stack(cluster_centers)
    # normalize cluster centers
    cluster_centers = F.normalize(cluster_centers, dim=1).float().to(device)

    # reinitialize classifier head
    features_dim = model.module.features_dim
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    model.module.head = nn.Linear(features_dim, num_clusters, bias=False).to(device)
    model.module.head.weight.data.copy_(cluster_centers)

    sampler = RandomMultipleGallerySampler(target_train_set, args.num_instances)
    train_target_loader = DataLoader(
        convert_to_pytorch_dataset(target_train_set, root=target_dataset.images_dir, transform=train_transform),
        batch_size=args.batch_size, num_workers=args.workers, sampler=sampler, pin_memory=True, drop_last=True)
    train_target_iter = ForeverDataIterator(train_target_loader)

    return train_target_iter, num_clusters


def train(train_target_iter: ForeverDataIterator, model, optimizer, criterion_ce: CrossEntropyLossWithLabelSmooth,
          criterion_triplet: SoftTripletLoss, epoch: int, args: argparse.Namespace):
    # train with pseudo labels
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_ce = AverageMeter('CeLoss', ':3.2f')
    losses_triplet = AverageMeter('TripletLoss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_ce, losses_triplet, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i in range(args.iters_per_epoch):
        x_t, _, labels_t, _ = next(train_target_iter)

        x_t = x_t.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_t, f_t = model(x_t)

        # cross entropy loss
        loss_ce = criterion_ce(y_t, labels_t)
        # triplet loss
        loss_triplet = criterion_triplet(f_t, f_t, labels_t)
        loss = loss_ce + loss_triplet * args.trade_off

        cls_acc = accuracy(y_t, labels_t)[0]
        losses_ce.update(loss_ce.item(), x_t.size(0))
        losses_triplet.update(loss_triplet.item(), x_t.size(0))
        losses.update(loss.item(), x_t.size(0))
        cls_accs.update(cls_acc.item(), x_t.size(0))

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
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )
    parser = argparse.ArgumentParser(description="Cluster Baseline for Domain Adaptative ReID")
    # dataset parameters
    parser.add_argument('source_root', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', type=str, help='source domain')
    parser.add_argument('-t', '--target', type=str, help='target domain')
    parser.add_argument('--train-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='reid_resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: reid_resnet50)')
    parser.add_argument('--num-clusters', type=int, default=500)
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--finetune', action='store_true', help='whether use 10x smaller lr for backbone')
    parser.add_argument('--rate', type=float, default=0.2)
    # training parameters
    parser.add_argument('--clustering-algorithm', type=str, default='dbscan', choices=['kmeans', 'dbscan'],
                        help='clustering algorithm to run, currently supported method: ["kmeans", "dbscan"]')
    parser.add_argument('--resume', type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument('--pretrained-model-path', type=str, help='path to pretrained (source-only) model')
    parser.add_argument('--trade-off', type=float, default=1,
                        help='trade-off hyper parameter between cross entropy loss and triplet loss')
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--iters-per-epoch', type=int, default=400)
    parser.add_argument('--print-freq', type=int, default=40)
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument('--rerank', action='store_true', help="evaluation only")
    parser.add_argument("--log", type=str, default='baseline_cluster',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
