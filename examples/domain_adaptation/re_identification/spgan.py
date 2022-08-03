"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import sys
import argparse
import itertools
import os.path as osp
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T

sys.path.append('../../..')
import tllib.translation.cyclegan as cyclegan
import tllib.translation.spgan as spgan
from tllib.translation.cyclegan.util import ImagePool, set_requires_grad
import tllib.vision.datasets.reid as datasets
from tllib.vision.datasets.reid.convert import convert_to_pytorch_dataset
from tllib.vision.transforms import Denormalize
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger

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
        T.Resize(args.load_size, Image.BICUBIC),
        T.RandomCrop(args.input_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    working_dir = osp.dirname(osp.abspath(__file__))
    root = osp.join(working_dir, args.root)

    source_dataset = datasets.__dict__[args.source](root=osp.join(root, args.source.lower()))
    train_source_loader = DataLoader(
        convert_to_pytorch_dataset(source_dataset.train, root=source_dataset.images_dir, transform=train_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True, drop_last=True)

    target_dataset = datasets.__dict__[args.target](root=osp.join(root, args.target.lower()))
    train_target_loader = DataLoader(
        convert_to_pytorch_dataset(target_dataset.train, root=target_dataset.images_dir, transform=train_transform),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True, drop_last=True)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # define networks (generators, discriminators and siamese network)
    netG_S2T = cyclegan.generator.__dict__[args.netG](ngf=args.ngf, norm=args.norm, use_dropout=False).to(device)
    netG_T2S = cyclegan.generator.__dict__[args.netG](ngf=args.ngf, norm=args.norm, use_dropout=False).to(device)
    netD_S = cyclegan.discriminator.__dict__[args.netD](ndf=args.ndf, norm=args.norm).to(device)
    netD_T = cyclegan.discriminator.__dict__[args.netD](ndf=args.ndf, norm=args.norm).to(device)
    siamese_net = spgan.SiameseNetwork(nsf=args.nsf).to(device)

    # create image buffer to store previously generated images
    fake_S_pool = ImagePool(args.pool_size)
    fake_T_pool = ImagePool(args.pool_size)

    # define optimizer and lr scheduler
    optimizer_G = Adam(itertools.chain(netG_S2T.parameters(), netG_T2S.parameters()), lr=args.lr,
                       betas=(args.beta1, 0.999))
    optimizer_D = Adam(itertools.chain(netD_S.parameters(), netD_T.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_siamese = Adam(siamese_net.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    lr_decay_function = lambda epoch: 1.0 - max(0, epoch - args.epochs) / float(args.epochs_decay)
    lr_scheduler_G = LambdaLR(optimizer_G, lr_lambda=lr_decay_function)
    lr_scheduler_D = LambdaLR(optimizer_D, lr_lambda=lr_decay_function)
    lr_scheduler_siamese = LambdaLR(optimizer_siamese, lr_lambda=lr_decay_function)

    # optionally resume from a checkpoint
    if args.resume:
        print("Resume from", args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')

        netG_S2T.load_state_dict(checkpoint['netG_S2T'])
        netG_T2S.load_state_dict(checkpoint['netG_T2S'])
        netD_S.load_state_dict(checkpoint['netD_S'])
        netD_T.load_state_dict(checkpoint['netD_T'])
        siamese_net.load_state_dict(checkpoint['siamese_net'])

        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        optimizer_siamese.load_state_dict(checkpoint['optimizer_siamese'])
        lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
        lr_scheduler_D.load_state_dict(checkpoint['lr_scheduler_D'])
        lr_scheduler_siamese.load_state_dict(checkpoint['lr_scheduler_siamese'])

        args.start_epoch = checkpoint['epoch'] + 1

    if args.phase == 'test':
        transform = T.Compose([
            T.Resize(args.test_input_size, Image.BICUBIC),
            cyclegan.transform.Translation(netG_S2T, device)
        ])
        source_dataset.translate(transform, osp.join(args.translated_root, args.source.lower()))
        return

    # define loss function
    criterion_gan = cyclegan.LeastSquaresGenerativeAdversarialLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_contrastive = spgan.ContrastiveLoss(margin=args.margin)

    # define visualization function
    tensor_to_image = T.Compose([
        Denormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        T.ToPILImage()
    ])

    def visualize(image, name):
        """
        Args:
            image (tensor): image in shape 3 x H x W
            name: name of the saving image
        """
        tensor_to_image(image).save(logger.get_image_path("{}.png".format(name)))

    # start training
    for epoch in range(args.start_epoch, args.epochs + args.epochs_decay):
        logger.set_epoch(epoch)
        print(lr_scheduler_G.get_lr())

        # train for one epoch
        train(train_source_iter, train_target_iter, netG_S2T, netG_T2S, netD_S, netD_T, siamese_net,
              criterion_gan, criterion_cycle, criterion_identity, criterion_contrastive,
              optimizer_G, optimizer_D, optimizer_siamese,
              fake_S_pool, fake_T_pool, epoch, visualize, args)

        # update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        lr_scheduler_siamese.step()

        # save checkpoint
        torch.save(
            {
                'netG_S2T': netG_S2T.state_dict(),
                'netG_T2S': netG_T2S.state_dict(),
                'netD_S': netD_S.state_dict(),
                'netD_T': netD_T.state_dict(),
                'siamese_net': siamese_net.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'optimizer_siamese': optimizer_siamese.state_dict(),
                'lr_scheduler_G': lr_scheduler_G.state_dict(),
                'lr_scheduler_D': lr_scheduler_D.state_dict(),
                'lr_scheduler_siamese': lr_scheduler_siamese.state_dict(),
                'epoch': epoch,
                'args': args
            }, logger.get_checkpoint_path(epoch)
        )

    if args.translated_root is not None:
        transform = T.Compose([
            T.Resize(args.test_input_size, Image.BICUBIC),
            cyclegan.transform.Translation(netG_S2T, device)
        ])
        source_dataset.translate(transform, osp.join(args.translated_root, args.source.lower()))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          netG_S2T, netG_T2S, netD_S, netD_T, siamese_net: spgan.SiameseNetwork,
          criterion_gan: cyclegan.LeastSquaresGenerativeAdversarialLoss,
          criterion_cycle: nn.L1Loss, criterion_identity: nn.L1Loss,
          criterion_contrastive: spgan.ContrastiveLoss,
          optimizer_G: Adam, optimizer_D: Adam, optimizer_siamese: Adam,
          fake_S_pool: ImagePool, fake_T_pool: ImagePool, epoch: int, visualize, args: argparse.Namespace):
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
    losses_contrastive_G = AverageMeter('contrastive_G', ':3.2f')
    losses_contrastive_siamese = AverageMeter('contrastive_siamese', ':3.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_G_S2T, losses_G_T2S, losses_D_S, losses_D_T,
         losses_cycle_S, losses_cycle_T, losses_identity_S, losses_identity_T,
         losses_contrastive_G, losses_contrastive_siamese],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()

    for i in range(args.iters_per_epoch):
        real_S, _, _, _ = next(train_source_iter)
        real_T, _, _, _ = next(train_target_iter)

        real_S = real_S.to(device)
        real_T = real_T.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # Compute fake images and reconstruction images.
        fake_T = netG_S2T(real_S)
        rec_S = netG_T2S(fake_T)
        fake_S = netG_T2S(real_T)
        rec_T = netG_S2T(fake_S)

        # ===============================================
        # train the generators (every two iterations)
        # ===============================================
        if i % 2 == 0:
            # save memory
            set_requires_grad(netD_S, False)
            set_requires_grad(netD_T, False)
            set_requires_grad(siamese_net, False)
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

            # siamese network output
            f_real_S = siamese_net(real_S)
            f_fake_T = siamese_net(fake_T)
            f_real_T = siamese_net(real_T)
            f_fake_S = siamese_net(fake_S)

            # positive pair
            loss_contrastive_p_G = criterion_contrastive(f_real_S, f_fake_T, 0) + \
                                   criterion_contrastive(f_real_T, f_fake_S, 0)
            # negative pair
            loss_contrastive_n_G = criterion_contrastive(f_fake_T, f_real_T, 1) + \
                                   criterion_contrastive(f_fake_S, f_real_S, 1) + \
                                   criterion_contrastive(f_real_S, f_real_T, 1)
            # contrastive loss
            loss_contrastive_G = (loss_contrastive_p_G + 0.5 * loss_contrastive_n_G) / 4 * args.trade_off_contrastive

            # combined loss and calculate gradients
            loss_G = loss_G_S2T + loss_G_T2S + loss_cycle_S + loss_cycle_T + loss_identity_S + loss_identity_T
            if epoch > 1:
                loss_G += loss_contrastive_G
            netG_S2T.zero_grad()
            netG_T2S.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # update corresponding statistics
            losses_G_S2T.update(loss_G_S2T.item(), real_S.size(0))
            losses_G_T2S.update(loss_G_T2S.item(), real_S.size(0))
            losses_cycle_S.update(loss_cycle_S.item(), real_S.size(0))
            losses_cycle_T.update(loss_cycle_T.item(), real_S.size(0))
            losses_identity_S.update(loss_identity_S.item(), real_S.size(0))
            losses_identity_T.update(loss_identity_T.item(), real_S.size(0))
            if epoch > 1:
                losses_contrastive_G.update(loss_contrastive_G, real_S.size(0))

        # ===============================================
        # train the siamese network (when epoch > 0)
        # ===============================================
        if epoch > 0:
            set_requires_grad(siamese_net, True)
            # siamese network output
            f_real_S = siamese_net(real_S)
            f_fake_T = siamese_net(fake_T.detach())
            f_real_T = siamese_net(real_T)
            f_fake_S = siamese_net(fake_S.detach())

            # positive pair
            loss_contrastive_p_siamese = criterion_contrastive(f_real_S, f_fake_T, 0) + \
                                         criterion_contrastive(f_real_T, f_fake_S, 0)
            # negative pair
            loss_contrastive_n_siamese = criterion_contrastive(f_real_S, f_real_T, 1)
            # contrastive loss
            loss_contrastive_siamese = (loss_contrastive_p_siamese + 2 * loss_contrastive_n_siamese) / 3

            # update siamese network
            siamese_net.zero_grad()
            loss_contrastive_siamese.backward()
            optimizer_siamese.step()

            # update corresponding statistics
            losses_contrastive_siamese.update(loss_contrastive_siamese, real_S.size(0))

        # ===============================================
        # train the discriminators
        # ===============================================

        set_requires_grad(netD_S, True)
        set_requires_grad(netD_T, True)
        # Calculate GAN loss for discriminator D_S
        fake_S_ = fake_S_pool.query(fake_S.detach())
        loss_D_S = 0.5 * (criterion_gan(netD_S(real_S), True) + criterion_gan(netD_S(fake_S_), False))
        # Calculate GAN loss for discriminator D_T
        fake_T_ = fake_T_pool.query(fake_T.detach())
        loss_D_T = 0.5 * (criterion_gan(netD_T(real_T), True) + criterion_gan(netD_T(fake_T_), False))

        # update discriminators
        netD_S.zero_grad()
        netD_T.zero_grad()
        loss_D_S.backward()
        loss_D_T.backward()
        optimizer_D.step()

        # update corresponding statistics
        losses_D_S.update(loss_D_S.item(), real_S.size(0))
        losses_D_T.update(loss_D_T.item(), real_S.size(0))

        # measure elapsed time
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
    parser = argparse.ArgumentParser(description='SPGAN for Domain Adaptative ReID')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-s', '--source', type=str, help='source domain')
    parser.add_argument('-t', '--target', type=str, help='target domain')
    parser.add_argument('--load-size', nargs='+', type=int, default=(286, 144), help='loading image size')
    parser.add_argument('--input-size', nargs='+', type=int, default=(256, 128),
                        help='the input and output image size during training')
    # model parameters
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--nsf', type=int, default=64, help='# of sianet filters int the first conv layer')
    parser.add_argument('--netD', type=str, default='patch',
                        help='specify discriminator architecture [patch | pixel]. The basic model is a 70x70 PatchGAN.')
    parser.add_argument('--netG', type=str, default='resnet_9',
                        help='specify generator architecture [resnet_9 | resnet_6 | unet_256 | unet_128]')
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization [instance | batch | none]')
    # training parameters
    parser.add_argument("--resume", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument('--trade-off-cycle', type=float, default=10.0, help='trade off for cycle loss')
    parser.add_argument('--trade-off-identity', type=float, default=5.0, help='trade off for identity loss')
    parser.add_argument('--trade-off-contrastive', type=float, default=2.0, help='trade off for contrastive loss')
    parser.add_argument('--margin', type=float, default=2,
                        help='margin for contrastive loss')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epochs-decay', type=int, default=15,
                        help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('-i', '--iters-per-epoch', default=2000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--pool-size', type=int, default=50,
                        help='the size of image buffer that stores previously generated images')
    parser.add_argument('-p', '--print-freq', default=500, type=int,
                        metavar='N', help='print frequency (default: 500)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='spgan',
                        help="Where to save logs, checkpoints and debugging images.")
    # test parameters
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--translated-root', type=str, default=None,
                        help="The root to put the translated dataset")
    parser.add_argument('--test-input-size', nargs='+', type=int, default=(256, 128),
                        help='the input image size during testing')
    args = parser.parse_args()
    main(args)
