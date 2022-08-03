"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import sys
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage

sys.path.append('../../..')
from tllib.alignment.regda import PoseResNet2d as RegDAPoseResNet, \
    PseudoLabelGenerator2d, RegressionDisparity
import tllib.vision.models as models
from tllib.vision.models.keypoint_detection.pose_resnet import Upsampling, PoseResNet
from tllib.vision.models.keypoint_detection.loss import JointsKLLoss
import tllib.vision.datasets.keypoint_detection as datasets
import tllib.vision.transforms.keypoint_detection as T
from tllib.vision.transforms import Denormalize
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter, AverageMeterDict
from tllib.utils.metric.keypoint_detection import accuracy
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
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.RandomRotation(args.rotation),
        T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        T.GaussianBlur(),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        normalize
    ])
    image_size = (args.image_size, args.image_size)
    heatmap_size = (args.heatmap_size, args.heatmap_size)
    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(root=args.source_root, transforms=train_transform,
                                          image_size=image_size, heatmap_size=heatmap_size)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_source_dataset = source_dataset(root=args.source_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_source_loader = DataLoader(val_source_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    target_dataset = datasets.__dict__[args.target]
    train_target_dataset = target_dataset(root=args.target_root, transforms=train_transform,
                                          image_size=image_size, heatmap_size=heatmap_size)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_target_dataset = target_dataset(root=args.target_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_target_loader = DataLoader(val_target_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print("Source train:", len(train_source_loader))
    print("Target train:", len(train_target_loader))
    print("Source test:", len(val_source_loader))
    print("Target test:", len(val_target_loader))

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    backbone = models.__dict__[args.arch](pretrained=True)
    upsampling = Upsampling(backbone.out_features)
    num_keypoints = train_source_dataset.num_keypoints
    model = RegDAPoseResNet(backbone, upsampling, 256, num_keypoints, num_head_layers=args.num_head_layers, finetune=True).to(device)
    # define loss function
    criterion = JointsKLLoss()
    pseudo_label_generator = PseudoLabelGenerator2d(num_keypoints, args.heatmap_size, args.heatmap_size)
    regression_disparity = RegressionDisparity(pseudo_label_generator, JointsKLLoss(epsilon=1e-7))

    # define optimizer and lr scheduler
    optimizer_f = SGD([
        {'params': backbone.parameters(), 'lr': 0.1},
        {'params': upsampling.parameters(), 'lr': 0.1},
    ], lr=0.1, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    optimizer_h = SGD(model.head.parameters(), lr=1., momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    optimizer_h_adv = SGD(model.head_adv.parameters(), lr=1., momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_decay_function = lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
    lr_scheduler_f = LambdaLR(optimizer_f, lr_decay_function)
    lr_scheduler_h = LambdaLR(optimizer_h, lr_decay_function)
    lr_scheduler_h_adv = LambdaLR(optimizer_h_adv, lr_decay_function)
    start_epoch = 0

    if args.resume is None:
        if args.pretrain is None:
            # first pretrain the backbone and upsampling
            print("Pretraining the model on source domain.")
            args.pretrain = logger.get_checkpoint_path('pretrain')
            pretrained_model = PoseResNet(backbone, upsampling, 256, num_keypoints, True).to(device)
            optimizer = SGD(pretrained_model.get_parameters(lr=args.lr), momentum=args.momentum, weight_decay=args.wd, nesterov=True)
            lr_scheduler = MultiStepLR(optimizer, args.lr_step, args.lr_factor)
            best_acc = 0
            for epoch in range(args.pretrain_epochs):
                lr_scheduler.step()
                print(lr_scheduler.get_lr())

                pretrain(train_source_iter, pretrained_model, criterion, optimizer, epoch, args)
                source_val_acc = validate(val_source_loader, pretrained_model, criterion, None, args)

                # remember best acc and save checkpoint
                if source_val_acc['all'] > best_acc:
                    best_acc = source_val_acc['all']
                    torch.save(
                        {
                            'model': pretrained_model.state_dict()
                        }, args.pretrain
                    )
                print("Source: {} best: {}".format(source_val_acc['all'], best_acc))

        # load from the pretrained checkpoint
        pretrained_dict = torch.load(args.pretrain, map_location='cpu')['model']
        model_dict = model.state_dict()
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        # optionally resume from a checkpoint
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer_f.load_state_dict(checkpoint['optimizer_f'])
        optimizer_h.load_state_dict(checkpoint['optimizer_h'])
        optimizer_h_adv.load_state_dict(checkpoint['optimizer_h_adv'])
        lr_scheduler_f.load_state_dict(checkpoint['lr_scheduler_f'])
        lr_scheduler_h.load_state_dict(checkpoint['lr_scheduler_h'])
        lr_scheduler_h_adv.load_state_dict(checkpoint['lr_scheduler_h_adv'])
        start_epoch = checkpoint['epoch'] + 1

    # define visualization function
    tensor_to_image = Compose([
        Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToPILImage()
    ])

    def visualize(image, keypoint2d, name, heatmaps=None):
        """
        Args:
            image (tensor): image in shape 3 x H x W
            keypoint2d (tensor): keypoints in shape K x 2
            name: name of the saving image
        """
        train_source_dataset.visualize(tensor_to_image(image),
                                       keypoint2d, logger.get_image_path("{}.jpg".format(name)))

    if args.phase == 'test':
        # evaluate on validation set
        source_val_acc = validate(val_source_loader, model, criterion, None, args)
        target_val_acc = validate(val_target_loader, model, criterion, visualize, args)
        print("Source: {:4.3f} Target: {:4.3f}".format(source_val_acc['all'], target_val_acc['all']))
        for name, acc in target_val_acc.items():
            print("{}: {:4.3f}".format(name, acc))
        return

    # start training
    best_acc = 0
    print("Start regression domain adaptation.")
    for epoch in range(start_epoch, args.epochs):
        logger.set_epoch(epoch)
        print(lr_scheduler_f.get_lr(), lr_scheduler_h.get_lr(), lr_scheduler_h_adv.get_lr())

        # train for one epoch
        train(train_source_iter, train_target_iter, model, criterion, regression_disparity,
              optimizer_f, optimizer_h, optimizer_h_adv, lr_scheduler_f, lr_scheduler_h, lr_scheduler_h_adv,
              epoch, visualize if args.debug else None, args)

        # evaluate on validation set
        source_val_acc = validate(val_source_loader, model, criterion, None, args)
        target_val_acc = validate(val_target_loader, model, criterion, visualize if args.debug else None, args)

        # remember best acc and save checkpoint
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer_f': optimizer_f.state_dict(),
                'optimizer_h': optimizer_h.state_dict(),
                'optimizer_h_adv': optimizer_h_adv.state_dict(),
                'lr_scheduler_f': lr_scheduler_f.state_dict(),
                'lr_scheduler_h': lr_scheduler_h.state_dict(),
                'lr_scheduler_h_adv': lr_scheduler_h_adv.state_dict(),
                'epoch': epoch,
                'args': args
            }, logger.get_checkpoint_path(epoch)
        )
        if target_val_acc['all'] > best_acc:
            shutil.copy(logger.get_checkpoint_path(epoch), logger.get_checkpoint_path('best'))
            best_acc = target_val_acc['all']
        print("Source: {:4.3f} Target: {:4.3f} Target(best): {:4.3f}".format(source_val_acc['all'], target_val_acc['all'], best_acc))
        for name, acc in target_val_acc.items():
            print("{}: {:4.3f}".format(name, acc))

    logger.close()


def pretrain(train_source_iter, model, criterion, optimizer,
             epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_s = AverageMeter('Loss (s)', ":.2e")
    acc_s = AverageMeter("Acc (s)", ":3.2f")

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_s, acc_s],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        optimizer.zero_grad()

        x_s, label_s, weight_s, meta_s = next(train_source_iter)

        x_s = x_s.to(device)
        label_s = label_s.to(device)
        weight_s = weight_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s = model(x_s)
        loss_s = criterion(y_s, label_s, weight_s)

        # compute gradient and do SGD step
        loss_s.backward()
        optimizer.step()

        # measure accuracy and record loss
        _, avg_acc_s, cnt_s, pred_s = accuracy(y_s.detach().cpu().numpy(),
                                               label_s.detach().cpu().numpy())
        acc_s.update(avg_acc_s, cnt_s)
        losses_s.update(loss_s, cnt_s)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train(train_source_iter, train_target_iter, model, criterion,regression_disparity,
          optimizer_f, optimizer_h, optimizer_h_adv, lr_scheduler_f, lr_scheduler_h, lr_scheduler_h_adv,
          epoch: int, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_s = AverageMeter('Loss (s)', ":.2e")
    losses_gf = AverageMeter('Loss (t, false)', ":.2e")
    losses_gt = AverageMeter('Loss (t, truth)', ":.2e")
    acc_s = AverageMeter("Acc (s)", ":3.2f")
    acc_t = AverageMeter("Acc (t)", ":3.2f")
    acc_s_adv = AverageMeter("Acc (s, adv)", ":3.2f")
    acc_t_adv = AverageMeter("Acc (t, adv)", ":3.2f")

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_s, losses_gf, losses_gt, acc_s, acc_t, acc_s_adv, acc_t_adv],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, label_s, weight_s, meta_s = next(train_source_iter)
        x_t, label_t, weight_t, meta_t = next(train_target_iter)

        x_s = x_s.to(device)
        label_s = label_s.to(device)
        weight_s = weight_s.to(device)

        x_t = x_t.to(device)
        label_t = label_t.to(device)
        weight_t = weight_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # Step A train all networks to minimize loss on source domain
        optimizer_f.zero_grad()
        optimizer_h.zero_grad()
        optimizer_h_adv.zero_grad()

        y_s, y_s_adv = model(x_s)
        loss_s = criterion(y_s, label_s, weight_s) + \
                 args.margin * args.trade_off * regression_disparity(y_s, y_s_adv, weight_s, mode='min')
        loss_s.backward()
        optimizer_f.step()
        optimizer_h.step()
        optimizer_h_adv.step()

        # Step B train adv regressor to maximize regression disparity
        optimizer_h_adv.zero_grad()
        y_t, y_t_adv = model(x_t)
        loss_ground_false = args.trade_off * regression_disparity(y_t, y_t_adv, weight_t, mode='max')
        loss_ground_false.backward()
        optimizer_h_adv.step()

        # Step C train feature extractor to minimize regression disparity
        optimizer_f.zero_grad()
        y_t, y_t_adv = model(x_t)
        loss_ground_truth = args.trade_off * regression_disparity(y_t, y_t_adv, weight_t, mode='min')
        loss_ground_truth.backward()
        optimizer_f.step()

        # do update step
        model.step()
        lr_scheduler_f.step()
        lr_scheduler_h.step()
        lr_scheduler_h_adv.step()

        # measure accuracy and record loss
        _, avg_acc_s, cnt_s, pred_s = accuracy(y_s.detach().cpu().numpy(),
                                               label_s.detach().cpu().numpy())
        acc_s.update(avg_acc_s, cnt_s)
        _, avg_acc_t, cnt_t, pred_t = accuracy(y_t.detach().cpu().numpy(),
                                               label_t.detach().cpu().numpy())
        acc_t.update(avg_acc_t, cnt_t)
        _, avg_acc_s_adv, cnt_s_adv, pred_s_adv = accuracy(y_s_adv.detach().cpu().numpy(),
                                               label_s.detach().cpu().numpy())
        acc_s_adv.update(avg_acc_s_adv, cnt_s)
        _, avg_acc_t_adv, cnt_t_adv, pred_t_adv = accuracy(y_t_adv.detach().cpu().numpy(),
                                               label_t.detach().cpu().numpy())
        acc_t_adv.update(avg_acc_t_adv, cnt_t)
        losses_s.update(loss_s, cnt_s)
        losses_gf.update(loss_ground_false, cnt_s)
        losses_gt.update(loss_ground_truth, cnt_s)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if visualize is not None:
                visualize(x_s[0], pred_s[0] * args.image_size / args.heatmap_size, "source_{}_pred".format(i))
                visualize(x_s[0], meta_s['keypoint2d'][0], "source_{}_label".format(i))
                visualize(x_t[0], pred_t[0] * args.image_size / args.heatmap_size, "target_{}_pred".format(i))
                visualize(x_t[0], meta_t['keypoint2d'][0], "target_{}_label".format(i))
                visualize(x_s[0], pred_s_adv[0] * args.image_size / args.heatmap_size, "source_adv_{}_pred".format(i))
                visualize(x_t[0], pred_t_adv[0] * args.image_size / args.heatmap_size, "target_adv_{}_pred".format(i))


def validate(val_loader, model, criterion, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    acc = AverageMeterDict(val_loader.dataset.keypoints_group.keys(), ":3.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc['all']],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, label, weight, meta) in enumerate(val_loader):
            x = x.to(device)
            label = label.to(device)
            weight = weight.to(device)

            # compute output
            y = model(x)
            loss = criterion(y, label, weight)

            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))
            acc_per_points, avg_acc, cnt, pred = accuracy(y.cpu().numpy(),
                                                          label.cpu().numpy())

            group_acc = val_loader.dataset.group_accuracy(acc_per_points)
            acc.update(group_acc, x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                if visualize is not None:
                    visualize(x[0], pred[0] * args.image_size / args.heatmap_size, "val_{}_pred.jpg".format(i))
                    visualize(x[0], meta['keypoint2d'][0], "val_{}_label.jpg".format(i))

    return acc.average()


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

    parser = argparse.ArgumentParser(description='RegDA for Keypoint Detection Domain Adaptation')
    # dataset parameters
    parser.add_argument('source_root', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--resize-scale', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--rotation', type=int, default=180,
                        help='rotation range of the RandomRotation augmentation')
    parser.add_argument('--image-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--heatmap-size', type=int, default=64,
                        help='output heatmap size')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet101)')
    parser.add_argument("--pretrain", type=str, default=None,
                        help="Where restore pretrained model parameters from.")
    parser.add_argument("--resume", type=str, default=None,
                        help="where restore model parameters from.")
    parser.add_argument('--num-head-layers', type=int, default=2)
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.0001, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-step', default=[45, 60], type=tuple, help='parameter for lr scheduler')
    parser.add_argument('--lr-factor', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--pretrain_epochs', default=70, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='regda',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--debug', action="store_true",
                        help='In the debug mode, save images and predictions')
    args = parser.parse_args()
    main(args)

