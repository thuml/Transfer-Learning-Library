"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import warnings
import argparse
import shutil
import sys

import math
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.append('../..')
from tllib.vision.datasets.segmentation import NYUv2
from tllib.utils.logger import CompleteLogger
from models import HardParameterSharingModel
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    logger = CompleteLogger(args.log, "train")
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
    train_dataset = NYUv2(root=args.root, mode="train", augmentation=True)
    val_dataset = NYUv2(root=args.root, mode="val", augmentation=False)
    test_dataset = NYUv2(root=args.root, mode='test', augmentation=False)
    print("train: {} val: {} test: {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    # create model
    num_out_channels = train_dataset.num_out_channels
    model = HardParameterSharingModel({
        task_name: num_out_channels[task_name]
        for task_name in args.train_tasks
    }).to(device)

    # define optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # start training
    base_result = {
        'segmentation': [0.5251, 0.7478],
        'depth': [0.4047, 0.1719],
        'normal': [22.6744, 15.9096, 0.3717, 0.6353, 0.7418]
    }
    indicator = {
        'segmentation': [0, 0],
        'depth': [1, 1],
        'normal': [1, 1, 0, 0, 0]
    }  # set to 0 if a higher value indicates better performance for the k-th task and otherwise 1
    best_improvement = -math.inf
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        utils.train(train_loader, model, optimizer, epoch, args, device)
        lr_scheduler.step()

        # evaluate on validation set
        result = utils.validate(val_loader, model, args, device)
        improvement = utils.count_improvement(base_result, result, indicator)
        # remember best acc@1 and save checkpoint
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
        if improvement > best_improvement:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_improvement = improvement

        print("best_improvement = {}\nimprovement = {}\nresult={}\n".format(best_improvement, improvement, result))

    # evaluate on test set
    model.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    result = utils.validate(test_loader, model, args, device)
    improvement = utils.count_improvement(base_result, result, indicator)
    print("test improvement = {}\nresult={}\n".format(improvement, result))

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NYUv2 ERM')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-tr', '--train-tasks', help='training tasks(s)', nargs='+')
    parser.add_argument('-ts', '--test-tasks', help='test task(s)', nargs='+')
    parser.add_argument('--img-size', type=int, default=(288, 384), nargs='+')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma for StepLR')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='erm',
                        help="Where to save logs, checkpoints and debugging images.")
    args = parser.parse_args()
    main(args)
