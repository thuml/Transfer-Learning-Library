"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import argparse
import os
import builtins
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("please install apex from https://www.github.com/nvidia/apex to run this example.")

import utils
from tllib.modules.classifier import Classifier as ClassifierBase
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.self_training.mean_teacher import EMATeacher
from tllib.self_training.dst import WorstCaseEstimationLoss
from tllib.vision.transforms import MultipleApply
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter
from tllib.utils.metric import accuracy


class Classifier(ClassifierBase):

    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)

        # pseudo head and worst-case estimation head
        mlp_dim = 2 * self.features_dim
        self.pseudo_head = nn.Sequential(
            nn.Linear(self.features_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_dim, self.num_classes)
        )
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=300000, auto_step=False)
        self.worst_head = nn.Sequential(
            nn.Linear(self.features_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_dim, self.num_classes)
        )

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        y = self.head(f)
        if not self.training:
            return y
        y_pseudo = self.pseudo_head(f)
        f_adv = self.grl_layer(f)
        y_adv = self.worst_head(f_adv)
        return y, y_adv, y_pseudo

    def get_parameters(self, lr, weight_decay):
        if self.finetune:
            raise NotImplementedError

        wd_params = []
        no_wd_params = []
        for name, param in self.named_parameters():
            if 'bias' in name or 'bn' in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        params = [
            {"params": wd_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_wd_params, "lr": lr, "weight_decay": 0.}
        ]

        return params

    def step(self):
        self.grl_layer.step()


def main(args):
    writer, logger = None, None
    if args.local_rank == 0:
        logger = CompleteLogger(args.log, args.phase)
        if args.phase == 'train':
            writer = SummaryWriter(args.log)
    else:
        def print_pass(*print_args):
            pass

        # suppress printing if not master
        builtins.print = print_pass

    print(args)
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "amp requires cudnn backend to be enabled."

    # data loading code
    weak_transform = utils.get_train_transform(
        img_size=args.img_size,
        random_horizontal_flip=True,
        rand_augment=False,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std
    )
    strong_transform = utils.get_train_transform(
        img_size=args.img_size,
        random_horizontal_flip=True,
        rand_augment=True,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std
    )
    labeled_train_transform = weak_transform
    unlabeled_train_transform = MultipleApply([weak_transform, strong_transform])

    val_transform = utils.get_val_transform(
        img_size=args.img_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std
    )

    print('labeled_train_transform: ', labeled_train_transform)
    print('unlabeled_train_transform: ', unlabeled_train_transform)
    print('val_transform: ', val_transform)

    labeled_train_dataset, unlabeled_train_dataset, val_dataset, args.num_classes = \
        utils.get_dataset(args.dataset, args.root, args.num_samples_per_class, args.batch_size * args.world_size,
                          labeled_train_transform, val_transform, unlabeled_train_transform=unlabeled_train_transform,
                          seed=args.seed)

    print('labeled_train_dataset_size: ', args.num_samples_per_class * args.num_classes)
    print('unlabeled_train_dataset_size: ', len(unlabeled_train_dataset))
    print('val_dataset_size: ', len(val_dataset))

    labeled_train_sampler = RandomSampler(labeled_train_dataset, replacement=True,
                                          num_samples=args.batch_size * args.world_size * args.train_iterations,
                                          generator=None)
    unlabeled_train_sampler = RandomSampler(unlabeled_train_dataset, replacement=True,
                                            num_samples=args.unlabeled_batch_size * args.world_size *
                                                        args.train_iterations, generator=None)
    val_sampler = None
    if args.distributed:
        labeled_train_sampler = utils.DistributedProxySampler(labeled_train_sampler)
        unlabeled_train_sampler = utils.DistributedProxySampler(unlabeled_train_sampler)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    labeled_train_loader = DataLoader(
        labeled_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
        sampler=labeled_train_sampler, drop_last=True)
    unlabeled_train_loader = DataLoader(
        unlabeled_train_dataset, batch_size=args.unlabeled_batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=False, sampler=unlabeled_train_sampler, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
        sampler=val_sampler)

    # create model
    print('=> creating model {}'.format(args.arch))
    backbone = utils.get_model(args.arch, depth=args.depth, widen_factor=args.widen_factor)
    model = Classifier(backbone, args.num_classes, finetune=False)
    print(model)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format
    model = model.cuda().to(memory_format=memory_format)

    # typical methods evaluate with ema model
    ema_model = EMATeacher(model, alpha=args.alpha)

    # define optimizer
    optimizer = torch.optim.SGD(model.get_parameters(args.lr, args.weight_decay), momentum=0.9, nesterov=True)

    # initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )
    # cosine lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: math.cos(7. / 16. * math.pi * x / args.train_iterations)
    )

    # wrap the model with apex.parallel.DistributedDataParallel
    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    if args.phase == 'test':
        # load checkpoint for evaluation
        assert args.checkpoint_path is not None
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        utils.update_bn(model, ema_model)
        acc = utils.validate(val_loader, ema_model, args)
        return

    args.start_step = 0
    # optionally resume from a checkpoint
    if args.resume:
        print('resume from checkpoint path {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        amp.load_state_dict(checkpoint['amp'])
        args.start_step = checkpoint['global_step'] + 1

    # start training
    train(labeled_train_loader, unlabeled_train_loader, val_loader, model, ema_model, optimizer, lr_scheduler, writer,
          logger, args)


def train(labeled_train_loader: DataLoader, unlabeled_train_loader: DataLoader, val_loader: DataLoader, model: DDP,
          ema_model: EMATeacher, optimizer: torch.optim.SGD, lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
          writer: SummaryWriter, logger: CompleteLogger, args: argparse.Namespace):
    print('training')
    best_acc = 0.

    batch_time = AverageMeter('time', ':3.1f')
    cls_losses = AverageMeter('cls loss', ':3.2f')
    cls_accs = AverageMeter('cls acc', ':3.2f')
    self_training_losses = AverageMeter('self training loss', ':3.2f')
    pseudo_label_ratios = AverageMeter('pseudo label ratio', ':3.2f')
    worst_case_losses = AverageMeter('worst loss', ':3.2f')
    losses = AverageMeter('loss', ':3.2f')

    # define loss function
    cls_criterion = utils.CrossEntropyLoss()
    self_training_criterion = utils.ConfidenceBasedSelfTrainingLoss(args.threshold)
    worst_case_criterion = WorstCaseEstimationLoss(eta_prime=args.eta_prime)

    # switch to train mode
    model.train()
    end = time.time()
    batch_size = (args.batch_size + args.unlabeled_batch_size) * args.world_size

    for global_step, (x_l, labels_l), ((x_u_weak, x_u_strong), _) in \
            zip(range(args.start_step, args.train_iterations), labeled_train_loader, unlabeled_train_loader):
        x_l = x_l.cuda()
        x_u_weak = x_u_weak.cuda()
        x_u_strong = x_u_strong.cuda()
        labels_l = labels_l.cuda()

        # compute output
        x = torch.cat((x_l, x_u_weak, x_u_strong), dim=0)
        y, y_adv, y_pseudo = model(x)

        # cls loss on labeled data
        y_l = y[:args.batch_size]
        cls_loss = cls_criterion(y_l, labels_l)

        # self training loss on unlabeled data
        y_u_weak, _ = y[args.batch_size:].chunk(2, dim=0)
        _, y_u_strong = y_pseudo[args.batch_size:].chunk(2, dim=0)
        self_training_loss, mask, _ = self_training_criterion(y_u_strong, y_u_weak)
        self_training_loss = args.trade_off_self_training * self_training_loss

        # worst case estimation loss on unlabeled data
        y_l_adv = y_adv[:args.batch_size]
        y_u_adv, _ = y_adv[args.batch_size:].chunk(2, dim=0)
        worst_case_loss = args.trade_off_worst * worst_case_criterion(y_l, y_l_adv, y_u_weak, y_u_adv)

        loss = cls_loss + self_training_loss + worst_case_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # update factor of gradient reverse layer
        utils.unwrap_ddp(model).step()

        # update ema model
        ema_model.update()
        utils.update_bn(model, ema_model)

        if global_step % args.print_freq == 0:
            # measure accuracy and track loss
            cls_acc, = accuracy(y_l.data, labels_l, topk=(1,))
            pseudo_label_ratio = mask.mean()
            if args.distributed:
                reduced_cls_loss = utils.reduce_tensor(cls_loss.data, args.world_size)
                reduced_self_training_loss = utils.reduce_tensor(self_training_loss.data, args.world_size)
                reduced_worst_case_loss = utils.reduce_tensor(worst_case_loss.data, args.world_size)
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                cls_acc = utils.reduce_tensor(cls_acc, args.world_size)
                pseudo_label_ratio = utils.reduce_tensor(pseudo_label_ratio, args.world_size)
            else:
                reduced_cls_loss = cls_loss.data
                reduced_self_training_loss = self_training_loss.data
                reduced_worst_case_loss = worst_case_loss.data
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            cls_losses.update(to_python_float(reduced_cls_loss), batch_size)
            self_training_losses.update(to_python_float(reduced_self_training_loss), batch_size)
            worst_case_losses.update(to_python_float(reduced_worst_case_loss), batch_size)
            losses.update(to_python_float(reduced_loss), batch_size)
            cls_accs.update(to_python_float(cls_acc), batch_size)
            pseudo_label_ratios.update(to_python_float(pseudo_label_ratio), batch_size)

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            print('iteration: [{0}]\t'
                  'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'speed {1:.3f} ({2:.3f})\t'
                  'cls loss {cls_loss.val:.10f} ({cls_loss.avg:.4f})\t'
                  'self training loss {self_training_loss.val:.10f} ({self_training_loss.avg:.4f})\t'
                  'worst case loss {worst_case_loss.val:.10f} ({worst_case_loss.avg:.4f})\t'
                  'loss {loss.val:.10f} ({loss.avg:.4f})\t'
                  'acc@1 {cls_accs.val:.3f} ({cls_accs.avg:.3f})\t'
                  'pseudo label ratio {ratio.val:.3f} ({ratio.avg:.3f})'.format(
                global_step, batch_size / batch_time.val, batch_size / batch_time.avg,
                batch_time=batch_time, cls_loss=cls_losses, self_training_loss=self_training_losses,
                worst_case_loss=worst_case_losses, loss=losses, cls_accs=cls_accs, ratio=pseudo_label_ratios))

            if args.local_rank == 0:
                writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('train/cls_loss', to_python_float(reduced_cls_loss), global_step)
                writer.add_scalar('train/self_training_loss', to_python_float(reduced_self_training_loss), global_step)
                writer.add_scalar('train/worst_case_loss', to_python_float(reduced_worst_case_loss), global_step)
                writer.add_scalar('train/loss', to_python_float(reduced_loss), global_step)
                writer.add_scalar('train/acc', to_python_float(cls_acc), global_step)
                writer.add_scalar('train/pseudo_label_ratio', to_python_float(pseudo_label_ratio), global_step)

                torchvision.utils.save_image(
                    x_l,
                    os.path.join(args.log, 'visualize', 'labeled-data.jpg'),
                    padding=0,
                    normalize=True)

                torchvision.utils.save_image(
                    x_u_weak,
                    os.path.join(args.log, 'visualize', 'weak-aug-unlabeled-data.jpg'),
                    padding=0,
                    normalize=True)

                torchvision.utils.save_image(
                    x_u_strong,
                    os.path.join(args.log, 'visualize', 'strong-aug-unlabeled-data.jpg'),
                    padding=0,
                    normalize=True)

        if global_step % args.eval_freq == 0:
            acc = utils.validate(val_loader, ema_model, args)
            if args.local_rank == 0:
                writer.add_scalar('eval/acc', acc, global_step)
                torch.save(
                    {
                        'model': model.state_dict(),
                        'ema_model': ema_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'amp': amp.state_dict(),
                        'global_step': global_step
                    }, logger.get_checkpoint_path('latest')
                )

                if acc > best_acc:
                    best_acc = acc
                    shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))

    print('best acc@1 {:.4f}'.format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DST (FixMatch)')

    # dataset parameters
    parser.add_argument('--root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()))
    parser.add_argument('--num-samples-per-class', default=4, type=int,
                        help='number of labeled samples per class')
    parser.add_argument('--img-size', default=(32, 32), type=int, metavar='N', nargs='+',
                        help='image patch size (default: 32 x 32)')
    parser.add_argument('--norm-mean', default=utils.CIFAR10_MEAN, type=float, nargs='+',
                        help='normalization mean')
    parser.add_argument('--norm-std', default=utils.CIFAR10_STD, type=float, nargs='+',
                        help='normalization std')

    # model parameters
    parser.add_argument('--arch', '-a', default='WideResNet', metavar='ARCH',
                        choices=utils.get_model_names(),
                        help='model architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: WideResNet)')
    parser.add_argument('--depth', default=28, type=int,
                        help='depth of backbone network')
    parser.add_argument('--widen-factor', default=2, type=int,
                        help='widen factor of backbone network')
    parser.add_argument('--channels-last', default=False, type=bool)

    # training parameters
    parser.add_argument('--checkpoint-path', default=None, type=str,
                        help='path to checkpoint in test phase')
    parser.add_argument('--resume', default=None, type=str,
                        help='where restore model parameters from')
    parser.add_argument('--alpha', default=0.999, type=float,
                        help='momentum for ema model update')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='confidence threshold')
    parser.add_argument('--trade-off-self-training', default=1, type=float,
                        help='weight of self-training loss on unlabeled data')
    parser.add_argument('--trade-off-worst', default=0.1, type=float,
                        help='weight of worst-case estimation loss on unlabeled data')
    parser.add_argument('--eta-prime', default=1, type=float,
                        help='weight of adversarial loss on labeled data')
    parser.add_argument('--train-iterations', default=1000000, type=int, metavar='N',
                        help='number of total iterations to run')
    parser.add_argument('--eval-freq', default=5000, type=int,
                        help='test interval length')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                        help='mini-batch size of labeled data per process')
    parser.add_argument('-ub', '--unlabeled-batch-size', default=448, type=int,
                        help='mini-batch size of unlabeled data per process')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training')

    # distributed training setups
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync-bn', action='store_true',
                        help='enabling apex sync BN')
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', action='store_true')
    parser.add_argument('--loss-scale', default=None, type=str)

    # log parameters
    parser.add_argument("--log", default='dst_fixmatch', type=str,
                        help="where to save logs, checkpoints and debugging images")
    parser.add_argument('--print-freq', '-p', default=2000, type=int, metavar='N',
                        help='print frequency (default: 2000)')
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")

    args = parser.parse_args()
    main(args)
