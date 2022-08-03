"""
Training a bounding box adaptor
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import os.path as osp
import argparse
from collections import deque
import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from detectron2.modeling.box_regression import Box2BoxTransform

from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.modules.regressor import Regressor
from tllib.alignment.mdd import ImageRegressor, RegressionMarginDisparityDiscrepancy
from tllib.alignment.d_adapt.proposal import ProposalDataset, PersistentProposalList, flatten, ExpandCrop

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BoxTransform(nn.Module):
    def __init__(self):
        super(BoxTransform, self).__init__()
        BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
        self.box_transform = Box2BoxTransform(weights=BBOX_REG_WEIGHTS)

    def forward(self, pred_delta, gt_classes, proposal_boxes):
        """
        Args:
            - pred_delta: predicted bounding box offset for each classes
            - gt_classes: ground truth classes
            - proposal_boxes: referenced bounding box

        Returns:
            predicted bounding box offset for ground truth classes
            and  predicted bounding box
        """
        gt_class_cols = 4 * gt_classes[:, None] + torch.arange(4, device=device)
        pred_delta = torch.gather(pred_delta, dim=1, index=gt_class_cols)
        pred_box = self.box_transform.apply_deltas(pred_delta, proposal_boxes)
        return pred_delta, pred_box


def iou_between(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
    reduction: str = "none"
):
    """Intersections over Union between two boxes"""
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    if reduction == 'mean':
        return iouk.mean()
    elif reduction == 'sum':
        return iouk.sum()
    else:
        return iouk


def clamp_single(box, w, h):
    x1, y1, x2, y2 = box
    x1 = x1.clamp(min=0, max=w)
    x2 = x2.clamp(min=0, max=w)
    y1 = y1.clamp(min=0, max=h)
    y2 = y2.clamp(min=0, max=h)
    return torch.tensor((x1, y1, x2, y2))


def clamp(boxes, widths, heights):
    """clamp (limit) the values in boxes within the widths and heights of the image."""
    clamped_boxes = []
    for box, w, h in zip(boxes, widths, heights):
        clamped_boxes.append(clamp_single(box, w, h))
    return torch.stack(clamped_boxes, dim=0)


class BoundingBoxAdaptor:
    def __init__(self, class_names, log, args):
        self.class_names = class_names
        for k, v in args._get_kwargs():
            setattr(args, k.replace("_b", ""), v)
        self.args = args
        print(self.args)
        self.logger = CompleteLogger(log)
        # create model
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = utils.get_model(args.arch, pretrain=not args.scratch)
        num_classes = len(class_names)
        bottleneck_dim = args.bottleneck_dim
        bottleneck = nn.Sequential(
            nn.Conv2d(backbone.out_features, bottleneck_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(),
        )
        head = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, num_classes * 4),
        )
        for layer in head:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)
        adv_head = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, num_classes * 4),
        )
        for layer in adv_head:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)
        self.model = ImageRegressor(
            backbone, num_classes * 4, bottleneck=bottleneck,
            head=head, adv_head=adv_head
        ).to(device)
        self.box_transform = BoxTransform()

    def load_checkpoint(self, path=None):
        if path is None:
            path = self.logger.get_checkpoint_path('latest')
        if osp.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            return True
        else:
            return False

    def prepare_training_data(self, proposal_list: PersistentProposalList, labeled=True):
        if not labeled:
            # remove (predicted) background proposals
            filtered_proposals_list = []
            for proposals in proposal_list:
                keep_indices = (0 <= proposals.pred_classes) & (proposals.pred_classes < len(self.class_names))
                filtered_proposals_list.append(proposals[keep_indices])
        else:
            # remove proposals with low IoU
            filtered_proposals_list = []
            for proposals in proposal_list:
                keep_indices = proposals.gt_ious > 0.3
                filtered_proposals_list.append(proposals[keep_indices])

        filtered_proposals_list = flatten(filtered_proposals_list, self.args.max_train)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize((self.args.resize_size, self.args.resize_size)),
            # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            # T.RandomGrayscale(),
            T.ToTensor(),
            normalize
        ])

        dataset = ProposalDataset(filtered_proposals_list, transform, crop_func=ExpandCrop(self.args.expand))
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=True, num_workers=self.args.workers, drop_last=True)
        return dataloader

    def prepare_validation_data(self, proposal_list: PersistentProposalList):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize((self.args.resize_size, self.args.resize_size)),
            T.ToTensor(),
            normalize
        ])

        # remove (predicted) background proposals
        filtered_proposals_list = []
        for proposals in proposal_list:
            # keep_indices = (0 <= proposals.gt_classes) & (proposals.gt_classes < len(self.class_names))
            keep_indices = (0 <= proposals.pred_classes) & (proposals.pred_classes < len(self.class_names))
            filtered_proposals_list.append(proposals[keep_indices])

        filtered_proposals_list = flatten(filtered_proposals_list, self.args.max_val)
        dataset = ProposalDataset(filtered_proposals_list, transform, crop_func=ExpandCrop(self.args.expand))
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.workers, drop_last=False)
        return dataloader

    def prepare_test_data(self, proposal_list: PersistentProposalList):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize((self.args.resize_size, self.args.resize_size)),
            T.ToTensor(),
            normalize
        ])

        dataset = ProposalDataset(proposal_list, transform, crop_func=ExpandCrop(self.args.expand))
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.workers, drop_last=False)
        return dataloader

    def predict(self, data_loader):
        # switch to evaluate mode
        self.model.eval()
        predictions = deque()

        with torch.no_grad():
            for images, labels in tqdm.tqdm(data_loader):
                images = images.to(device)
                pred_classes = labels['pred_classes'].to(device)
                pred_boxes = labels['pred_boxes'].to(device).float()
                # compute output
                pred_deltas = self.model(images)
                _, pred_boxes = self.box_transform(pred_deltas, pred_classes, pred_boxes)
                pred_boxes = clamp(pred_boxes.cpu(), labels['width'], labels['height'])
                pred_boxes = pred_boxes.numpy().tolist()
                for p in pred_boxes:
                    predictions.append(p)
        return predictions

    def validate_baseline(self, val_loader):
        """call this function if you have labeled data for validation"""
        ious = AverageMeter("IoU", ":.4e")
        print("Calculate baseline IoU:")
        for _, labels in tqdm.tqdm(val_loader):
            gt_boxes = labels['gt_boxes']
            pred_boxes = labels['pred_boxes']
            ious.update(iou_between(pred_boxes, gt_boxes).mean().item(), gt_boxes.size(0))

        print(' * Baseline IoU {:.3f}'.format(ious.avg))
        return ious.avg

    @staticmethod
    def validate(val_loader, model, box_transform, args) -> float:
        """call this function if you have labeled data for validation"""
        batch_time = AverageMeter('Time', ':6.3f')
        ious = AverageMeter("IoU", ":.4e")
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, ious],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                pred_classes = labels['pred_classes'].to(device)
                gt_boxes = labels['gt_boxes'].to(device).float()
                pred_boxes = labels['pred_boxes'].to(device).float()

                # compute output
                pred_deltas = model(images)
                _, pred_boxes = box_transform(pred_deltas, pred_classes, pred_boxes)
                pred_boxes = clamp(pred_boxes.cpu(), labels['width'], labels['height'])
                ious.update(iou_between(pred_boxes, gt_boxes.cpu()).mean().item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            print(' * IoU {:.3f}'.format(ious.avg))

        return ious.avg

    def fit(self, data_loader_source, data_loader_target, data_loader_validation=None):
        """When no labels exists on target domain, please set data_loader_validation=None"""
        args = self.args
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

        iter_source = ForeverDataIterator(data_loader_source)
        iter_target = ForeverDataIterator(data_loader_target)


        best_iou = 0.
        box_transform = self.box_transform

        # first pre-train on the source domain
        model = Regressor(
            self.model.backbone, len(self.class_names) * 4,
            bottleneck=nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            ),
            head=nn.Linear(self.model.backbone.out_features, len(self.class_names) * 4),
            bottleneck_dim=self.model.backbone.out_features
        ).to(device)
        optimizer = Adam(model.get_parameters(), args.pretrain_lr, weight_decay=args.pretrain_weight_decay)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.pretrain_lr * (1. + args.pretrain_lr_gamma * float(x)) ** (-args.pretrain_lr_decay))

        for epoch in range(args.pretrain_epochs):
            print("lr:", lr_scheduler.get_last_lr()[0])
            batch_time = AverageMeter('Time', ':3.1f')
            data_time = AverageMeter('Data', ':3.1f')
            losses = AverageMeter('Loss', ':3.2f')
            ious = AverageMeter("IoU", ":.4e")
            progress = ProgressMeter(
                args.iters_per_epoch,
                [batch_time, data_time, losses, ious],
                prefix="Epoch: [{}]".format(epoch))

            # switch to train mode
            model.train()

            end = time.time()
            for i in range(args.iters_per_epoch):
                x_s, labels_s = next(iter_source)
                x_s = x_s.to(device)
                # bounding box offsets
                delta_s = box_transform.box_transform.get_deltas(labels_s['pred_boxes'], labels_s['gt_boxes']).to(device).float()
                pred_boxes_s = labels_s['pred_boxes'].to(device).float()
                gt_classes_s = labels_s['gt_fg_classes'].to(device)
                gt_boxes_s = labels_s['gt_boxes'].to(device).float()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                pred_delta_s, _ = model(x_s)
                pred_delta_s, pred_boxes_s = box_transform(pred_delta_s, gt_classes_s, pred_boxes_s)
                reg_loss = F.smooth_l1_loss(pred_delta_s, delta_s)
                loss = reg_loss

                losses.update(loss.item(), x_s.size(0))
                ious.update(iou_between(pred_boxes_s.cpu(), gt_boxes_s.cpu()).mean().item(), x_s.size(0))

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

            # evaluate on validation set
            if data_loader_validation is not None:
                iou = self.validate(data_loader_validation, model, box_transform, args)
                best_iou = max(iou, best_iou)

        # training on both domains
        model = self.model
        optimizer = SGD(model.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

        for epoch in range(args.epochs):
            print("lr:", lr_scheduler.get_last_lr()[0])
            # train for one epoch
            batch_time = AverageMeter('Time', ':3.1f')
            data_time = AverageMeter('Data', ':3.1f')
            losses = AverageMeter('Loss', ':3.2f')
            ious = AverageMeter("IoU", ":.4e")
            ious_t = AverageMeter("IoU (t)", ":.4e")
            ious_s_adv = AverageMeter("IoU (s, adv)", ":.4e")
            ious_t_adv = AverageMeter("IoU (t, adv)", ":.4e")
            trans_losses = AverageMeter('Trans Loss', ':3.2f')
            progress = ProgressMeter(
                args.iters_per_epoch,
                [batch_time, data_time, losses, trans_losses, ious, ious_t, ious_s_adv, ious_t_adv],
                prefix="Epoch: [{}]".format(epoch))
            # switch to train mode
            model.train()
            mdd = RegressionMarginDisparityDiscrepancy(args.margin).to(device)

            end = time.time()
            for i in range(args.iters_per_epoch):
                x_s, labels_s = next(iter_source)
                x_t, labels_t = next(iter_target)
                x_s = x_s.to(device)
                x_t = x_t.to(device)
                # bounding box offsets
                delta_s = box_transform.box_transform.get_deltas(labels_s['pred_boxes'], labels_s['gt_boxes']).to(device).float()
                pred_boxes_s = labels_s['pred_boxes'].to(device).float()
                gt_classes_s = labels_s['gt_fg_classes'].to(device)
                gt_boxes_s = labels_s['gt_boxes'].to(device).float()
                pred_boxes_t = labels_t['pred_boxes'].to(device).float()
                gt_classes_t = labels_t['pred_classes'].to(device)
                gt_boxes_t = labels_t['gt_boxes'].to(device).float()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                x = torch.cat([x_s, x_t], dim=0)
                outputs, outputs_adv = model(x)
                pred_delta_s, pred_delta_t = outputs.chunk(2, dim=0)
                pred_delta_s_adv, pred_delta_t_adv = outputs_adv.chunk(2, dim=0)
                pred_delta_s, pred_boxes_s = box_transform(pred_delta_s, gt_classes_s, pred_boxes_s)
                pred_delta_t, pred_boxes_t = box_transform(pred_delta_t, gt_classes_t, pred_boxes_t)
                pred_delta_s_adv, pred_boxes_s_adv = box_transform(pred_delta_s_adv, gt_classes_s, pred_boxes_s)
                pred_delta_t_adv, pred_boxes_t_adv = box_transform(pred_delta_t_adv, gt_classes_t, pred_boxes_t)

                reg_loss = F.smooth_l1_loss(pred_delta_s, delta_s)
                # compute margin disparity discrepancy between domains
                transfer_loss = mdd(pred_delta_s, pred_delta_s_adv, pred_delta_t, pred_delta_t_adv)
                # for adversarial classifier, minimize negative mdd is equal to maximize mdd
                loss = reg_loss - transfer_loss * args.trade_off
                model.step()

                losses.update(loss.item(), x_s.size(0))
                ious.update(iou_between(pred_boxes_s.cpu(), gt_boxes_s.cpu()).mean().item(), x_s.size(0))
                ious_t.update(iou_between(pred_boxes_t.cpu(), gt_boxes_t.cpu()).mean().item(), x_s.size(0))
                ious_s_adv.update(iou_between(pred_boxes_s_adv.cpu(), gt_boxes_s.cpu()).mean().item(), x_s.size(0))
                ious_t_adv.update(iou_between(pred_boxes_t_adv.cpu(), gt_boxes_t.cpu()).mean().item(), x_s.size(0))
                trans_losses.update(transfer_loss.item(), x_s.size(0))

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

            # evaluate on validation set
            if data_loader_validation is not None:
                iou = self.validate(data_loader_validation, model, box_transform, args)
                best_iou = max(iou, best_iou)

            # save checkpoint
            torch.save(model.state_dict(), self.logger.get_checkpoint_path('latest'))

        print("best_iou = {:3.1f}".format(best_iou))

        self.logger.logger.flush()

    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        # dataset parameters
        parser.add_argument('--resize-size-b', type=int, default=224,
                            help='the image size after resizing')
        parser.add_argument('--max-train-b', type=int, default=10)
        parser.add_argument('--max-val-b', type=int, default=10)
        parser.add_argument('--expand-b', type=float, default=2.,
                            help='The expanding ratio between the input of the bounding box adaptor'
                                 '(the crops of objects) and the the original predicted box.')
        # model parameters
        parser.add_argument('--arch-b', metavar='ARCH', default='resnet101',
                            choices=utils.get_model_names(),
                            help='backbone architecture: ' +
                                 ' | '.join(utils.get_model_names()) +
                                 ' (default: resnet101)')
        parser.add_argument('--bottleneck-dim-b', default=1024, type=int,
                            help='Dimension of bottleneck')
        parser.add_argument('--no-pool-b', action='store_true',
                            help='no pool layer after the feature extractor.')
        parser.add_argument('--scratch-b', action='store_true', help='whether train from scratch.')
        parser.add_argument('--margin', type=float, default=4., help="margin hyper-parameter")
        parser.add_argument('--trade-off', default=0.1, type=float,
                            help='the trade-off hyper-parameter for transfer loss')
        # training parameters
        parser.add_argument('--batch-size-b', default=32, type=int,
                            metavar='N',
                            help='mini-batch size (default: 64)')
        parser.add_argument('--lr-b', default=0.004, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--lr-gamma-b', default=0.0002, type=float, help='parameter for lr scheduler')
        parser.add_argument('--lr-decay-b', default=0.75, type=float, help='parameter for lr scheduler')
        parser.add_argument('--weight-decay-b', default=5e-4, type=float,
                            metavar='W', help='weight decay (default: 5e-4)')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        parser.add_argument('--workers-b', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 2)')
        parser.add_argument('--epochs-b', default=2, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--pretrain-lr-b', default=0.001, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--pretrain-lr-gamma-b', default=0.0002, type=float, help='parameter for lr scheduler')
        parser.add_argument('--pretrain-lr-decay-b', default=0.75, type=float, help='parameter for lr scheduler')
        parser.add_argument('--pretrain-weight-decay-b', default=1e-3, type=float,
                            metavar='W', help='weight decay (default: 1e-3)')
        parser.add_argument('--pretrain-epochs-b', default=10, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--iters-per-epoch-b', default=1000, type=int,
                            help='Number of iterations per epoch')
        parser.add_argument('--print-freq-b', default=100, type=int,
                            metavar='N', help='print frequency (default: 100)')
        parser.add_argument('--seed-b', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument("--log-b", type=str, default='box',
                            help="Where to save logs, checkpoints and debugging images.")
        return parser


