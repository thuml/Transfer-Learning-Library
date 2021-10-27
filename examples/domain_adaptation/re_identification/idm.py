import random
import time
import warnings
import sys
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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN

sys.path.append('../../..')
import dalib.adaptation.idm.models as models
from dalib.adaptation.idm.models.identifier import ReIdentifier
from dalib.adaptation.idm.xbm import XBM
from dalib.adaptation.idm.loss import BridgeFeatLoss, BridgeProbLoss, DivLoss
from dalib.adaptation.idm.utils import filter_layers, convert_dsbn_idm
import common.vision.datasets.reid as datasets
from common.vision.datasets.reid.convert import convert_to_pytorch_dataset
from common.utils.metric.reid import extract_reid_feature, validate, visualize_ranked_results
from common.vision.models.reid.loss import TripletLoss, TripletLossXBM
from common.utils.data import ForeverDataIterator, RandomMultipleGallerySampler
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
    sampler = RandomMultipleGallerySampler(source_dataset.train, args.num_instances)
    train_source_loader = DataLoader(
        convert_to_pytorch_dataset(source_dataset.train, root=source_dataset.images_dir, transform=train_transform),
        batch_size=args.batch_size, num_workers=args.workers, sampler=sampler, pin_memory=True, drop_last=True)
    train_source_iter = ForeverDataIterator(train_source_loader)
    cluster_source_loader = DataLoader(
        convert_to_pytorch_dataset(source_dataset.train, root=source_dataset.images_dir, transform=val_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    val_loader = DataLoader(
        convert_to_pytorch_dataset(list(set(source_dataset.query) | set(source_dataset.gallery)),
                                   root=source_dataset.images_dir, transform=val_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    # target dataset
    target_dataset = datasets.__dict__[args.target](root=osp.join(target_root, args.target.lower()))
    cluster_target_loader = DataLoader(
        convert_to_pytorch_dataset(target_dataset.train, root=target_dataset.images_dir, transform=val_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        convert_to_pytorch_dataset(list(set(target_dataset.query) | set(target_dataset.gallery)),
                                   root=target_dataset.images_dir, transform=val_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    n_s_classes = source_dataset.num_train_pids
    args.n_classes = n_s_classes + len(target_dataset.train)
    args.n_s_classes = n_s_classes
    args.n_t_classes = len(target_dataset.train)

    # create model
    backbone = models.__dict__[args.arch](pretrained=True)
    pool_layer = nn.Identity() if args.no_pool else None
    model = ReIdentifier(backbone, args.n_classes, finetune=args.finetune, pool_layer=pool_layer)
    features_dim = model.features_dim

    idm_bn_names = filter_layers(args.stage)
    convert_dsbn_idm(model, idm_bn_names, idm=False)

    model = model.to(device)
    model = DataParallel(model)

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
        print("Test on target domain:")
        validate(test_loader, model, target_dataset.query, target_dataset.gallery, device, cmc_flag=True,
                 rerank=args.rerank)
        return

    # create XBM
    dataset_size = len(source_dataset.train) + len(target_dataset.train)
    memory_size = int(args.ratio * dataset_size)
    xbm = XBM(memory_size, features_dim)

    # initialize source-domain class centroids
    source_feature_dict = extract_reid_feature(cluster_source_loader, model, device, normalize=True)
    source_features_per_id = {}
    for f, pid, _ in source_dataset.train:
        if pid not in source_features_per_id:
            source_features_per_id[pid] = []
        source_features_per_id[pid].append(source_feature_dict[f].unsqueeze(0))
    source_centers = [torch.cat(source_features_per_id[pid], 0).mean(0) for pid in
                      sorted(source_features_per_id.keys())]
    source_centers = torch.stack(source_centers, 0)
    source_centers = F.normalize(source_centers, dim=1)
    model.module.head.weight.data[0: n_s_classes].copy_(source_centers.to(device))

    # save memory
    del source_centers, cluster_source_loader, source_features_per_id

    # define optimizer and lr scheduler
    optimizer = Adam(model.module.get_parameters(base_lr=args.lr, rate=args.rate), args.lr,
                     weight_decay=args.weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        utils.copy_state_dict(model, checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    # start training
    best_test_mAP = 0.
    for epoch in range(args.start_epoch, args.epochs):
        # run clustering algorithm and generate pseudo labels
        train_target_iter = run_dbscan(cluster_target_loader, model, target_dataset, train_transform, args)

        # train for one epoch
        print(lr_scheduler.get_lr())
        train(train_source_iter, train_target_iter, model, optimizer, xbm, epoch, args)

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            # remember best mAP and save checkpoint
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, logger.get_checkpoint_path(epoch)
            )
            print("Test on target domain...")
            _, test_mAP = validate(test_loader, model, target_dataset.query, target_dataset.gallery, device,
                                   cmc_flag=True, rerank=args.rerank)
            if test_mAP > best_test_mAP:
                shutil.copy(logger.get_checkpoint_path(epoch), logger.get_checkpoint_path('best'))
            best_test_mAP = max(test_mAP, best_test_mAP)

        # update lr
        lr_scheduler.step()

    print("best mAP on target = {}".format(best_test_mAP))
    logger.close()


def run_dbscan(cluster_loader: DataLoader, model: DataParallel, target_dataset, train_transform,
               args: argparse.Namespace):
    # run dbscan clustering algorithm
    feature_dict = extract_reid_feature(cluster_loader, model, device, normalize=True)
    feature = torch.stack(list(feature_dict.values())).cpu()
    rerank_dist = utils.compute_rerank_dist(feature).numpy()

    print('Clustering with dbscan algorithm')
    dbscan = DBSCAN(eps=args.eps, min_samples=4, metric='precomputed', n_jobs=-1)
    cluster_labels = dbscan.fit_predict(rerank_dist)
    print('Clustering finished')

    # generate training set with pseudo labels and calculate cluster centers
    target_train_set = []
    cluster_centers = {}
    for i, ((fname, _, cid), label) in enumerate(zip(target_dataset.train, cluster_labels)):
        if label == -1:
            continue
        target_train_set.append((fname, args.n_s_classes + label, cid))

        if label not in cluster_centers:
            cluster_centers[label] = []
        cluster_centers[label].append(feature[i])

    cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
    cluster_centers = torch.stack(cluster_centers)
    # normalize cluster centers
    cluster_centers = F.normalize(cluster_centers, dim=1).float().to(device)

    # reinitialize classifier head
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    args.n_t_classes = num_clusters
    model.module.head.weight.data[args.n_s_classes: args.n_s_classes + args.n_t_classes].copy_(cluster_centers)

    sampler = RandomMultipleGallerySampler(target_train_set, args.num_instances)
    train_target_loader = DataLoader(
        convert_to_pytorch_dataset(target_train_set, root=target_dataset.images_dir, transform=train_transform),
        batch_size=args.batch_size, num_workers=args.workers, sampler=sampler, pin_memory=True, drop_last=True)
    train_target_iter = ForeverDataIterator(train_target_loader)

    return train_target_iter


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model, optimizer: Adam,
          xbm: XBM, epoch: int, args: argparse.Namespace):
    # define loss function
    criterion_ce = BridgeProbLoss(args.n_s_classes + args.n_t_classes).to(device)
    criterion_triplet = TripletLoss(margin=args.margin).to(device)
    criterion_triplet_xbm = TripletLossXBM(margin=args.margin).to(device)
    criterion_bridge_feat = BridgeFeatLoss().to(device)
    criterion_diverse = DivLoss().to(device)

    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_ce = AverageMeter('CeLoss', ':3.2f')
    losses_triplet = AverageMeter('TripletLoss', ':3.2f')
    losses_triplet_xbm = AverageMeter('XBMTripletLoss', ':3.2f')
    losses_bridge_prob = AverageMeter('BridgeProbLoss', ':3.2f')
    losses_bridge_feat = AverageMeter('BridgeFeatLoss', ':3.2f')
    losses_diverse = AverageMeter('DiverseLoss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')

    cls_accs_s = AverageMeter('Src Cls Acc', ':3.1f')
    cls_accs_t = AverageMeter('Tgt Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_ce, losses_triplet, losses_triplet_xbm, losses_bridge_prob, losses_bridge_feat,
         losses_diverse, losses, cls_accs_s, cls_accs_t],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i in range(args.iters_per_epoch):
        x_s, _, labels_s, _ = next(train_source_iter)
        x_t, _, labels_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # arrange batch for domain-specific BN
        device_num = torch.cuda.device_count()
        B, C, H, W = x_s.size()

        def reshape(tensor):
            return tensor.view(device_num, -1, C, H, W)

        x_s, x_t = reshape(x_s), reshape(x_t)
        x = torch.cat((x_s, x_t), 1).view(-1, C, H, W)

        labels = torch.cat((labels_s.view(device_num, -1), labels_t.view(device_num, -1)), 1)
        labels = labels.view(-1)

        # compute output
        y, f, attention_lam = model(x, stage=args.stage)
        y = y[:, 0:args.n_s_classes + args.n_t_classes]  # only (n_s_classes + n_t_classes) classes are meaningful

        # split feats
        ori_f = f.view(device_num, -1, f.size(-1))
        f_s, f_t, f_mixed = ori_f.split(ori_f.size(1) // 3, dim=1)
        ori_f = torch.cat((f_s, f_t), 1).view(-1, ori_f.size(-1))

        # cross entropy loss
        loss_ce, loss_bridge_prob = criterion_ce(y, labels, attention_lam[:, 0].detach(), device_num)
        # triplet loss
        loss_triplet = criterion_triplet(ori_f, labels)
        # diverse loss
        loss_diverse = criterion_diverse(attention_lam)
        # bridge feature loss
        f_s = f_s.contiguous().view(-1, f.size(-1))
        f_t = f_t.contiguous().view(-1, f.size(-1))
        f_mixed = f_mixed.contiguous().view(-1, f.size(-1))
        loss_bridge_feat = criterion_bridge_feat(f_s, f_t, f_mixed, attention_lam)
        # xbm triplet loss
        xbm.enqueue_dequeue(ori_f.detach(), labels.detach())
        xbm_f, xbm_labels = xbm.get()
        loss_triplet_xbm = criterion_triplet_xbm(ori_f, labels, xbm_f, xbm_labels)

        loss = (1. - args.mu1) * loss_ce + loss_triplet + loss_triplet_xbm + \
               args.mu1 * loss_bridge_prob + args.mu2 * loss_bridge_feat + args.mu3 * loss_diverse

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ori_y = y.view(device_num, -1, y.size(-1))
        y_s, y_t, _ = ori_y.split(ori_y.size(1) // 3, dim=1)
        cls_acc_s = accuracy(y_s.reshape(-1, y_s.size(-1)), labels_s)[0]
        cls_acc_t = accuracy(y_t.reshape(-1, y_t.size(-1)), labels_t)[0]

        # update statistics
        losses_ce.update(loss_ce.item(), x_s.size(0))
        losses_triplet.update(loss_triplet.item(), x_s.size(0))
        losses_triplet_xbm.update(loss_triplet_xbm.item(), x_s.size(0))
        losses_bridge_prob.update(loss_bridge_prob.item(), x_s.size(0))
        losses_bridge_feat.update(loss_bridge_feat.item(), x_s.size(0))
        losses_diverse.update(loss_diverse.item(), x_s.size(0))
        losses.update(loss.item(), x_s.size(0))

        cls_accs_s.update(cls_acc_s.item(), x_s.size(0))
        cls_accs_t.update(cls_acc_t.item(), x_s.size(0))

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
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    parser = argparse.ArgumentParser(description="IDM for Domain Adaptative ReID")
    # dataset parameters
    parser.add_argument('source_root', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', type=str, help='source domain')
    parser.add_argument('-t', '--target', type=str, help='target domain')
    parser.add_argument('--train-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='reid_resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: reid_resnet50)')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--finetune', action='store_true', help='whether use 10x smaller lr for backbone')
    parser.add_argument('--rate', type=float, default=0.2)
    parser.add_argument('--n-classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--n-s-classes', type=int, default=1000, help="number of source classes")
    parser.add_argument('--n-t-classes', type=int, default=1000, help="number of target classes")
    # training parameters
    parser.add_argument('--resume', type=str, default=None, help="Where to restore model parameters from.")
    parser.add_argument('--eps', type=float, default=0.6, help="max neighbor distance for DBSCAN")
    parser.add_argument('--margin', type=float, default=0.3, help='margin for triplet loss')
    parser.add_argument('--mu1', type=float, default=0.7, help="weight for prediction bridge loss")
    parser.add_argument('--mu2', type=float, default=0.1, help="weight for feature bridge loss")
    parser.add_argument('--mu3', type=float, default=1, help="weight for diverse loss")
    parser.add_argument('--ratio', type=float, default=1, help='ratio of dataset to store in memory bank')
    parser.add_argument('--stage', type=int, default=0, choices=range(5))
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--iters-per-epoch', type=int, default=400)
    parser.add_argument('--print-freq', type=int, default=40)
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument('--rerank', action='store_true', help="evaluation only")
    parser.add_argument("--log", type=str, default='idm',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
