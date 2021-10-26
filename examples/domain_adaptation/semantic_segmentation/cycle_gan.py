"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import sys
import argparse
import itertools

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, Compose

sys.path.append('../../..')
import dalib.translation.cyclegan as cyclegan
from dalib.translation.cyclegan.util import ImagePool, set_requires_grad
import common.vision.datasets.segmentation as datasets
from common.vision.transforms import Denormalize
import common.vision.transforms.segmentation as T
from common.utils.data import ForeverDataIterator
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
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
    train_transform = T.Compose([
        T.RandomResizedCrop(size=args.train_size, ratio=args.resize_ratio, scale=(0.5, 1.)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(root=args.source_root, transforms=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    target_dataset = datasets.__dict__[args.target]
    train_target_dataset = target_dataset(root=args.target_root, transforms=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # define networks (both generators and discriminators)
    netG_S2T = cyclegan.generator.__dict__[args.netG](ngf=args.ngf, norm=args.norm, use_dropout=False).to(device)
    netG_T2S = cyclegan.generator.__dict__[args.netG](ngf=args.ngf, norm=args.norm, use_dropout=False).to(device)
    netD_S = cyclegan.discriminator.__dict__[args.netD](ndf=args.ndf, norm=args.norm).to(device)
    netD_T = cyclegan.discriminator.__dict__[args.netD](ndf=args.ndf, norm=args.norm).to(device)

    # create image buffer to store previously generated images
    fake_S_pool = ImagePool(args.pool_size)
    fake_T_pool = ImagePool(args.pool_size)

    # define optimizer and lr scheduler
    optimizer_G = Adam(itertools.chain(netG_S2T.parameters(), netG_T2S.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = Adam(itertools.chain(netD_S.parameters(), netD_T.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
    lr_decay_function = lambda epoch: 1.0 - max(0, epoch - args.epochs) / float(args.epochs_decay)
    lr_scheduler_G = LambdaLR(optimizer_G, lr_lambda=lr_decay_function)
    lr_scheduler_D = LambdaLR(optimizer_D, lr_lambda=lr_decay_function)

    # optionally resume from a checkpoint
    if args.resume:
        print("Resume from", args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        netG_S2T.load_state_dict(checkpoint['netG_S2T'])
        netG_T2S.load_state_dict(checkpoint['netG_T2S'])
        netD_S.load_state_dict(checkpoint['netD_S'])
        netD_T.load_state_dict(checkpoint['netD_T'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
        lr_scheduler_D.load_state_dict(checkpoint['lr_scheduler_D'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.phase == 'test':
        transform = T.Compose([
            T.Resize(image_size=args.test_input_size),
            T.wrapper(cyclegan.transform.Translation)(netG_S2T, device),
        ])
        train_source_dataset.translate(transform, args.translated_root)
        return

    # define loss function
    criterion_gan = cyclegan.LeastSquaresGenerativeAdversarialLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # define visualization function
    tensor_to_image = Compose([
        Denormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToPILImage()
    ])

    def visualize(image, name):
        """
        Args:
            image (tensor): image in shape 3 x H x W
            name: name of the saving image
        """
        tensor_to_image(image).save(logger.get_image_path("{}.png".format(name)))

    # start training
    for epoch in range(args.start_epoch, args.epochs+args.epochs_decay):
        logger.set_epoch(epoch)
        print(lr_scheduler_G.get_lr())

        # train for one epoch
        train(train_source_iter, train_target_iter, netG_S2T, netG_T2S, netD_S, netD_T,
              criterion_gan, criterion_cycle, criterion_identity, optimizer_G, optimizer_D,
              fake_S_pool, fake_T_pool, epoch, visualize, args)

        # update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        # save checkpoint
        torch.save(
            {
                'netG_S2T': netG_S2T.state_dict(),
                'netG_T2S': netG_T2S.state_dict(),
                'netD_S': netD_S.state_dict(),
                'netD_T': netD_T.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'lr_scheduler_G': lr_scheduler_G.state_dict(),
                'lr_scheduler_D': lr_scheduler_D.state_dict(),
                'epoch': epoch,
                'args': args
            }, logger.get_checkpoint_path(epoch)
        )

    if args.translated_root is not None:
        transform = T.Compose([
            T.Resize(image_size=args.test_input_size),
            T.wrapper(cyclegan.transform.Translation)(netG_S2T, device),
        ])
        train_source_dataset.translate(transform, args.translated_root)

    logger.close()


def train(train_source_iter, train_target_iter, netG_S2T, netG_T2S, netD_S, netD_T,
          criterion_gan, criterion_cycle, criterion_identity, optimizer_G, optimizer_D,
          fake_S_pool, fake_T_pool, epoch: int, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_G_S2T = AverageMeter('G_S2T', ':3.2f')
    losses_G_T2S = AverageMeter('G_T2S', ':3.2f')
    losses_D_S = AverageMeter('D_S', ':3.2f')
    losses_D_T = AverageMeter('D_T', ':3.2f')
    losses_cycle_S = AverageMeter('cycle_S', ':3.2f')
    losses_cycle_T = AverageMeter('cycle_T', ':3.2f')
    losses_identity_S = AverageMeter('idt_S', ':3.2f')
    losses_identity_T = AverageMeter('idt_T', ':3.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_G_S2T, losses_G_T2S, losses_D_S, losses_D_T,
         losses_cycle_S, losses_cycle_T, losses_identity_S, losses_identity_T],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()

    for i in range(args.iters_per_epoch):
        real_S, _ = next(train_source_iter)
        real_T, _ = next(train_target_iter)

        real_S = real_S.to(device)
        real_T = real_T.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # Compute fake images and reconstruction images.
        fake_T = netG_S2T(real_S)
        rec_S = netG_T2S(fake_T)
        fake_S = netG_T2S(real_T)
        rec_T = netG_S2T(fake_S)

        # Optimizing generators
        # discriminators require no gradients
        set_requires_grad(netD_S, False)
        set_requires_grad(netD_T, False)

        optimizer_G.zero_grad()
        # GAN loss D_T(G_S2T(S))
        loss_G_S2T = criterion_gan(netD_T(fake_T), real=True)
        # GAN loss D_S(G_T2S(B))
        loss_G_T2S = criterion_gan(netD_S(fake_S), real=True)
        # Cycle loss || G_T2S(G_S2T(S)) - S||
        loss_cycle_S = criterion_cycle(rec_S, real_S) * args.trade_off_cycle
        # Cycle loss || G_S2T(G_T2S(T)) - T||
        loss_cycle_T = criterion_cycle(rec_T, real_T) * args.trade_off_cycle
        # Identity loss
        # G_S2T should be identity if real_T is fed: ||G_S2T(real_T) - real_T||
        identity_T = netG_S2T(real_T)
        loss_identity_T = criterion_identity(identity_T, real_T) * args.trade_off_identity
        # G_T2S should be identity if real_S is fed: ||G_T2S(real_S) - real_S||
        identity_S = netG_T2S(real_S)
        loss_identity_S = criterion_identity(identity_S, real_S) * args.trade_off_identity
        # combined loss and calculate gradients
        loss_G = loss_G_S2T + loss_G_T2S + loss_cycle_S + loss_cycle_T + loss_identity_S + loss_identity_T
        loss_G.backward()
        optimizer_G.step()

        # Optimize discriminator
        set_requires_grad(netD_S, True)
        set_requires_grad(netD_T, True)
        optimizer_D.zero_grad()
        # Calculate GAN loss for discriminator D_S
        fake_S_ = fake_S_pool.query(fake_S.detach())
        loss_D_S = 0.5 * (criterion_gan(netD_S(real_S), True) + criterion_gan(netD_S(fake_S_), False))
        loss_D_S.backward()
        # Calculate GAN loss for discriminator D_T
        fake_T_ = fake_T_pool.query(fake_T.detach())
        loss_D_T = 0.5 * (criterion_gan(netD_T(real_T), True) + criterion_gan(netD_T(fake_T_), False))
        loss_D_T.backward()
        optimizer_D.step()

        # measure elapsed time
        losses_G_S2T.update(loss_G_S2T.item(), real_S.size(0))
        losses_G_T2S.update(loss_G_T2S.item(), real_S.size(0))
        losses_D_S.update(loss_D_S.item(), real_S.size(0))
        losses_D_T.update(loss_D_T.item(), real_S.size(0))
        losses_cycle_S.update(loss_cycle_S.item(), real_S.size(0))
        losses_cycle_T.update(loss_cycle_T.item(), real_S.size(0))
        losses_identity_S.update(loss_identity_S.item(), real_S.size(0))
        losses_identity_T.update(loss_identity_T.item(), real_S.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            for tensor, name in zip([real_S, real_T, fake_S, fake_T, rec_S, rec_T, identity_S, identity_T],
                                    ["real_S", "real_T", "fake_S", "fake_T", "rec_S",
                                     "rec_T", "identity_S", "identity_T"]):
                visualize(tensor[0], "{}_{}".format(i, name))


if __name__ == '__main__':
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )
    parser = argparse.ArgumentParser(description='CycleGAN for Segmentation Domain Adaptation')
    # dataset parameters
    parser.add_argument('source_root', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--resize-ratio', nargs='+', type=float, default=(1.5, 8 / 3.),
                        help='the resize ratio for the random resize crop')
    parser.add_argument('--train-size', nargs='+', type=int, default=(1024, 512),
                        help='the input and output image size during training')
    # model parameters
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='patch',
                        help='specify discriminator architecture [patch | pixel]. The basic model is a 70x70 PatchGAN.')
    parser.add_argument('--netG', type=str, default='unet_256',
                        help='specify generator architecture [resnet_9 | resnet_6 | unet_256 | unet_128]')
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument("--resume", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument('--trade-off-cycle', type=float, default=10.0, help='trade off for cycle loss')
    parser.add_argument('--trade-off-identity', type=float, default=5.0, help='trade off for identity loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epochs-decay', type=int, default=20,
                        help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('-i', '--iters-per-epoch', default=5000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--pool-size', type=int, default=50,
                        help='the size of image buffer that stores previously generated images')
    parser.add_argument('-p', '--print-freq', default=400, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='cyclegan',
                        help="Where to save logs, checkpoints and debugging images.")
    # test parameters
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--translated-root', type=str, default=None,
                        help="The root to put the translated dataset")
    parser.add_argument('--test-input-size', nargs='+', type=int, default=(1024, 512),
                        help='the input image size during test')
    args = parser.parse_args()
    main(args)
