from common.utils.analysis import collect_feature_and_labels, tsne, post_hoc_accuracy
from common.utils.logger import CompleteLogger
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.data import ForeverDataIterator
from common.vision.transforms import ResizeImage
import common.vision.models as models
from common.vision.datasets.checkerboard_officehome_211_split import CheckerboardOfficeHome211
# from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mdann import MultidomainAdversarialLoss, ImageClassifier
# from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.multidomain_discriminator import MultidomainDiscriminator
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utils_ajay import temp_scaling, kernel_ece
import wandb

sys.path.append('../../..')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_conf_mat(pred: List[int], y_true: List[int], 
    class_labels: str, folder_path: str, title: str, 
    normalize: Optional[bool]=None, 
    figsize: Optional[Tuple[int]]=(15,12)):
    conf_mat = confusion_matrix(y_true, pred, normalize=normalize)
    df_conf_mat = pd.DataFrame(conf_mat, index=class_labels, columns=class_labels)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df_conf_mat.to_csv(f'{folder_path}/{title}.csv')
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    sn.heatmap(df_conf_mat, annot=True)
    plt.savefig(f'{folder_path}/{title}.png')


def main(args: argparse.Namespace):
    # wandb login and intialize

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
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if args.center_crop:
        train_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), normalize
        ])
    else:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), normalize
        ])
    val_transform = T.Compose(
        [ResizeImage(256),
         T.CenterCrop(224),
         T.ToTensor(), normalize])

    # assuming that args.sources and arg.targets are disjoint sets
    # num_domains = len(args.sources) + len(args.targets)

    # TODO: create the train, val, test, novel dataset
    transforms_list = [
        train_transform, val_transform, val_transform
    ]

    datasets = CheckerboardOfficeHome211(
        root=args.root,
        download=False,
        balance_domains=args.balance_domains,
        transforms=transforms_list)

    # display the category-style matrix
    print(datasets)

    train_loader = DataLoader(datasets.train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              drop_last=True)

    val_loader = DataLoader(datasets.val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=True)

    novel_loader = DataLoader(datasets.novel_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              drop_last=True)

    if not args.use_forever_iter:
        args.iters_per_epoch = len(train_loader)
    wandb.login()
    wandb.init(project=args.wandb_name, config=args)

    train_iter = ForeverDataIterator(train_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(backbone,
                                 len(datasets.classes()),
                                 bottleneck_dim=args.bottleneck_dim).to(device)

    multidomain_discri = MultidomainDiscriminator(
        in_feature=classifier.features_dim,
        hidden_size=1024,
        num_domains=len(datasets.domains())).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() +
                    multidomain_discri.get_parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(
        optimizer, lambda x: args.lr *
        (1. + args.lr_gamma * float(x))**(-args.lr_decay))

    # define loss function for domain discrimination
    multidomain_adv = MultidomainAdversarialLoss(multidomain_discri).to(device)

    # resume from the best or latest checkpoint
    if args.phase != 'train':
        if args.use_best_model:
            checkpoint = torch.load(logger.get_checkpoint_path('best'),
                                    map_location='cpu')
        else:
            checkpoint = torch.load(logger.get_checkpoint_path('latest'),
                                    map_location='cpu')
        classifier.load_state_dict(checkpoint)

    def domain_analyze(features: torch.Tensor, domain_labels: torch.Tensor, dataset_type: str):
        title = f'{dataset_type} Dataset TSNE'
        tSNE_filename = osp.join(logger.visualize_directory, f'{title}.png')
        tsne.multidomain_visualize(
            features=features,
            domain_labels=domain_labels,
            filename=tSNE_filename,
            num_domains=len(datasets.domains()),
            fig_title=title
        )
        print("Saving t-SNE to", tSNE_filename)
        model = MultidomainDiscriminator(
            in_feature=classifier.features_dim,
            hidden_size=1024,
            num_domains=len(datasets.domains())).to(device)
        # calculate the accuracy of a model trained to discriminate the domains using the feature representation
        post_hoc_domain_acc, y_true, y_preds = post_hoc_accuracy.calculate_multidomain_acc(
                                                    model,
                                                    features,
                                                    domain_labels,
                                                    device
                                                )
        title = f'Post-Hoc Domain Discrimination Confusion Matrix ({dataset_type} Set)'
        styles = ['Art', 'Clipart', 'Product', 'Real World']
        generate_conf_mat(y_preds, y_true, styles, f'{args.log}/conf_mat/', title)
        print(f'Post-Hoc Domain Discrimination Accuracy ({dataset_type} Set) = {post_hoc_domain_acc}')
        return {f'Post-Hoc Domain Discrimination Accuracy ({dataset_type} Set)': post_hoc_domain_acc}


    def full_analysis(model: ImageClassifier):
        # extract features from train, validation, test, and novel dataset
        feature_extractor = nn.Sequential(model.backbone,
                                          model.bottleneck).to(device)
        train_features, train_labels = collect_feature_and_labels(
            train_loader, feature_extractor, device)
        train_domain_labels = CheckerboardOfficeHome211.get_style(train_labels)
        val_features, val_labels = collect_feature_and_labels(
            val_loader, feature_extractor, device)
        val_domain_labels = CheckerboardOfficeHome211.get_style(val_labels)
        novel_features, novel_labels = collect_feature_and_labels(
            novel_loader, feature_extractor, device)
        novel_domain_labels = CheckerboardOfficeHome211.get_style(novel_labels)
        all_features = torch.cat([train_features, val_features, novel_features], axis=0)
        all_domain_labels = torch.cat([train_domain_labels, val_domain_labels, novel_domain_labels], axis=0)
        
        log = {}
        # plot t-SNE and calculate post-hoc domain discriminator accuracy
        log.update(domain_analyze(train_features, train_domain_labels, 'Train'))
        log.update(domain_analyze(val_features, val_domain_labels, 'Validation'))
        log.update(domain_analyze(novel_features, novel_domain_labels, 'Novel'))
        log.update(domain_analyze(all_features, all_domain_labels, 'Total'))
        return log

    def partial_analysis(data_loader: DataLoader, dataset_type: str):
        feature_extractor = nn.Sequential(classifier.backbone,
                                        classifier.bottleneck).to(device)
        features, labels = collect_feature_and_labels(data_loader, 
                                        feature_extractor, device)
        domain_labels = CheckerboardOfficeHome211.get_style(labels)
        return domain_analyze(features, domain_labels, dataset_type)  

    # analysis the model
    if args.phase == 'analysis':
        wandb.log(full_analysis(classifier))
        wandb.finish()
        return

    if args.phase == 'novel':
        acc1, novel_log = validate(novel_loader, classifier, multidomain_adv,
                                   args, 'Novel', True)
        print(f'novel_acc1 = {acc1}')
        novel_log.update(partial_analysis(novel_log, 'Novel'))
        wandb.log(novel_log)
        wandb.finish()
        return

    # start training
    best_acc1 = 0.
    best_epoch = 0
    early_stop_count = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train_log = train(train_iter, classifier, multidomain_adv, optimizer,
                          lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1, val_log = validate(val_loader, classifier, multidomain_adv, args,
                                 'Validation', epoch == args.epochs - 1)

        total_log = train_log.copy()
        total_log.update(val_log.copy())
        wandb.log(total_log)

        # remember best acc@1 and save checkpoint
        # torch.save(classifier.state_dict(),
        #            logger.get_checkpoint_path('latest'))
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path(f'epoch {epoch}'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path(f'epoch {epoch}'),
                        logger.get_checkpoint_path('best'))
            best_epoch = epoch
        
        
        best_acc1 = max(acc1, best_acc1)
        

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    if args.use_best_model:
        classifier.load_state_dict(
            torch.load(logger.get_checkpoint_path('best')))
    else:
        # TODO: Fix to go after the latest model
        classifier.load_state_dict(
            torch.load(logger.get_checkpoint_path('latest')))

    # evaluate on novel set
    acc1, novel_log = validate(novel_loader, classifier, multidomain_adv, args,
                               'Novel', True)
    print("novel_acc1 = {:3.1f}".format(acc1))
    
    eval_log = {"Best Category Classification Accuracy (Validation Set)": best_acc1,
                "Epoch of Best Validation Accuracy (Validation Set)": best_epoch}
    eval_log.update(novel_log.copy())
    eval_log.update(full_analysis(classifier))
    wandb.log(eval_log)

    wandb.finish()
    logger.close()



def train(train_iter: ForeverDataIterator, model: ImageClassifier,
          multidomain_adv: MultidomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    total_losses = AverageMeter('Loss', ':6.2f')
    cls_losses = AverageMeter('Cls Loss', ':6.2f')
    transfer_losses = AverageMeter('Transfer Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, total_losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # define number of classes to predict

    # switch to train mode
    model.train()
    multidomain_adv.train()

    end = time.time()

    for i in range(args.iters_per_epoch):
        x_tr, labels_tr = next(train_iter)

        # retrieve the class and domain from the checkerboard office_home dataset
        class_labels_tr = CheckerboardOfficeHome211.get_category(labels_tr)
        domain_labels_tr = CheckerboardOfficeHome211.get_style(labels_tr)

        # add training data to device
        x_tr = x_tr.to(device)

        # add new labels to device
        class_labels_tr = class_labels_tr.to(device)
        domain_labels_tr = domain_labels_tr.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_tr, f_tr = model(x_tr)

        # calculate losses
        cls_loss = F.cross_entropy(y_tr, class_labels_tr)
        transfer_loss = multidomain_adv(f_tr, domain_labels_tr)
        total_loss = cls_loss + transfer_loss * args.trade_off
        
        if i % (1 + args.d_steps_per_g) < args.d_steps_per_g:
            loss_to_minimize = transfer_loss
        else:
            loss_to_minimize = total_loss

        # calculate accuracy
        cls_acc = accuracy(y_tr, class_labels_tr)[0]
        domain_acc = multidomain_adv.domain_discriminator_accuracy

        # update loss meter
        cls_losses.update(cls_loss.item(), x_tr.size(0))
        transfer_losses.update(transfer_loss.item(), x_tr.size(0))
        total_losses.update(total_loss.item(), x_tr.size(0))

        # update accuracy meters
        cls_accs.update(cls_acc.item(), x_tr.size(0))
        domain_accs.update(domain_acc.item(), x_tr.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_to_minimize.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # wandb
    return {
        "Category Classification Loss (Training Set)": cls_losses.sum,
        'Style Discrimination Loss (Training Set)': transfer_losses.sum,
        'Total Loss (Training Set)': total_losses.sum,
        'Category Classification Accuracy (Training Set)': cls_accs.avg,
        'Style Discrimination Accuracy (Training Set)': domain_accs.avg,
        'Classification Logits': y_tr,
    }


def validate(val_loader: DataLoader,
             model: ImageClassifier,
             multidomain_adv: MultidomainAdversarialLoss,
             args: argparse.Namespace,
             dataset_type: str,
             gen_conf_mat: Optional[bool] = False) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    cls_losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    transfer_losses = AverageMeter('Transfer Loss', ':6.2f')
    domain_accs = AverageMeter('Domain Acc', ':6.2f')
    losses = AverageMeter('Total Loss', ':6.2f')

    progress = ProgressMeter(len(val_loader),
                             [batch_time, cls_losses, top1, top5, domain_accs],
                             prefix=f'{dataset_type} Set: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    class_y_true = []
    class_preds = []
    domain_y_true = []
    domain_preds = []
    all_class_logits = []
    all_class_labels = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            class_labels = CheckerboardOfficeHome211.get_category(target)
            class_labels = class_labels.to(device)
            domain_labels = CheckerboardOfficeHome211.get_style(target)
            domain_labels = domain_labels.to(device)

            # compute output
            class_pred, features = model(images)

            # compute loss
            cls_loss = F.cross_entropy(class_pred, class_labels)
            transfer_loss = multidomain_adv(features, domain_labels)
            loss = cls_loss + transfer_loss * args.trade_off

            # measure accuracy and record class loss
            acc1, acc5 = accuracy(class_pred, class_labels, topk=(1, 5))
            if confmat:
                confmat.update(class_labels, class_pred.argmax(1))
            cls_losses.update(cls_loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # domain discrimination accuracy
            domain_acc = multidomain_adv.domain_discriminator_accuracy
            transfer_losses.update(transfer_loss.item(), images.size(0))
            domain_accs.update(domain_acc.item(), images.size(0))

            # gather data for calibration evaluation
            all_class_logits.append(class_pred)
            all_class_labels.append(class_labels)

            # gather data for the category confusion matrix
            _, y_tr_pred_class = class_pred.topk(1)
            class_y_true.append(class_labels)
            class_preds.append(y_tr_pred_class)

            # gather data for the category confusion matrix
            _, y_tr_pred_domain = multidomain_adv.domain_pred.topk(1)
            domain_y_true.append(domain_labels)
            domain_preds.append(y_tr_pred_domain)

            # record total loss on meter
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))
        if confmat:
            print(confmat.format(classes))

    # calculate expected calibration error
    all_class_logits = torch.cat(all_class_logits, dim=0)
    all_class_probs = F.softmax(all_class_logits, dim=1).tolist()
    all_class_labels = torch.squeeze(
        torch.cat(all_class_labels, dim=0)).tolist()
    classes = list(range(len(CheckerboardOfficeHome211.CATEGORIES)))
    ece = kernel_ece(all_class_probs, all_class_labels, classes)
    print(f"Expected Calibration Error: {ece}")

    if gen_conf_mat:
        class_y_true = torch.squeeze(torch.cat(class_y_true, dim=0)).tolist()
        class_preds = torch.squeeze(torch.cat(class_preds,
                                                    dim=0)).tolist()
        domain_y_true = torch.squeeze(torch.cat(domain_y_true, dim=0)).tolist()
        domain_preds = torch.squeeze(torch.cat(domain_preds,
                                                     dim=0)).tolist()
        styles = ['Art', 'Clipart', 'Product', 'Real World']
        cats = val_loader.dataset.classes
        # class_conf_mat = confusion_matrix(class_y_true, class_predicitons)
        # domain_conf_mat = confusion_matrix(domain_y_true, domain_predictions)
        folderpath = f'{args.log}/conf_mat'
        class_title = f'Category Classification Confusion Matrix ({dataset_type} Set)'
        domain_title = f'Style Discrimination Confusion Matrix ({dataset_type} Set)'
        
        # save the csv of the confusion matrix
        generate_conf_mat(class_preds, class_y_true, cats, folderpath, class_title, figsize=(25,22))
        generate_conf_mat(domain_preds, domain_y_true, styles, folderpath, domain_title)

    return top1.avg, {
        f"Category Classification Loss ({dataset_type} Set)": cls_losses.sum,
        f'Category Classification Accuracy ({dataset_type} Set)': top1.avg,
        f"Style Discriminator Loss ({dataset_type} Set)": transfer_losses.sum,
        f'Style Discriminator Accuracy ({dataset_type} Set)': domain_accs.avg,
        f"Total Loss ({dataset_type} Set)": losses.sum,
        f"Expected Calibration Error ({dataset_type} Set)": ece,
    }


if __name__ == '__main__':
    architecture_names = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(
        description='DANN for Checkerboard Domain Adapation on the Office-Home Dataset')
    # dataset parameters
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument('--center-crop',
                        default=False,
                        action='store_true',
                        help='whether use center crop during training')
    # model parameters
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                        ' | '.join(architecture_names) +
                        ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim',
                        default=256,
                        type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--trade-off',
                        default=1.,
                        type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b',
                        '--batch-size',
                        default=32,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.01,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--lr-gamma',
                        default=0.001,
                        type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--lr-decay',
                        default=0.75,
                        type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-3,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',
                        default=30,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 30)')
    parser.add_argument('-i',
                        '--iters-per-epoch',
                        default=1000,
                        type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p',
                        '--print-freq',
                        default=100,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument(
        '--per-class-eval',
        action='store_true',
        help='whether output per-class accuracy during evaluation')
    parser.add_argument(
        "--log",
        type=str,
        default='dann',
        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--wandb-name",
                        type=str,
                        default='checkerboard-task',
                        help="Name that will appear in the wandb dashboard.")
    parser.add_argument(
        "--phase",
        type=str,
        default='train',
        choices=['train', 'test', 'analysis', 'novel'],
        help="When phase is 'test', only test the model on test set."
        "When phase is 'novel', only test the model on the novel set."
        "When phase is 'analysis', only analysis the model.")
    # parser.add_argument('--balance-domains',
    #                     default=False,
    #                     action='store_true',
    #                     help='Balance the domains when creating category-style matrix.')
    parser.add_argument(
        '--balanced-domains',
        dest="balance_domains",
        default=True,
        action='store_true',
        help='''Balance the domains when creating category-style matrix.''')
    parser.add_argument(
        '--imbalanced-domains',
        dest="balance_domains",
        default=True,
        action='store_false',
        help='''Don't try to balance the domains when creating the
                category-style matrix.''')
    # parser.add_argument('--use-forever-iter',
    #                     default=False,
    #                     action='store_true',
    #                     help='Use the forever data iterator while training.')
    parser.add_argument('--forever-iter',
                        dest="use_forever_iter",
                        default=False,
                        action='store_true',
                        help='Use the forever data iterator while training.')
    parser.add_argument(
        '--no-forever-iter',
        dest="use_forever_iter",
        default=False,
        action='store_false',
        help="Don't use the forever data iterator while training.")
    parser.add_argument(
        '--use-best-model',
        dest="use_best_model",
        default=True,
        action='store_true',
        help='''If true, load the best model when testing and analyzing.
        If false, load the latest model when testing and analyzing.''')
    parser.add_argument(
        '--use-latest-model',
        dest="use_best_model",
        default=True,
        action='store_false',
        help='''If true, load the best model when testing and analyzing.
        If false, load the latest model when testing and analyzing.''')
    # parser.add_argument('--early-stopping',
    #                     default=False,
    #                     action='store_true',
    #                     help='use early stopping on validation set when training.')
    # parser.add_argument('--patience',
    #                     default=10,
    #                     type=int,
    #                     help='number of epochs to wait before employing earlystopping')
    parser.add_argument('--d-steps-per-g',
                        default=0,
                        type=int,
                        help='Number times the domain discriminator learns before the classifier starts learning.')

    args = parser.parse_args()
    main(args)
