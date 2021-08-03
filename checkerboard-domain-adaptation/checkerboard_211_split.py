from common.utils.analysis import collect_feature_and_labels, tsne, post_hoc_accuracy
from common.utils.logger import CompleteLogger
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.data import ForeverDataIterator
from common.vision.transforms import ResizeImage
import common.vision.models as models
from common.vision.datasets.checkerboard_officehome import CheckerboardOfficeHome
from dalib.adaptation.mdann import MultidomainAdversarialLoss, ImageClassifier
from dalib.modules.grl import WarmStartGradientReverseLayer, GradientReverseLayer
from dalib.modules.gl import GradientLayer, WarmStartGradientLayer

from dalib.modules.multidomain_discriminator import MultidomainDiscriminator
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import numpy as np
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
from utils_ajay import temp_scaling, kernel_ece, kernel_ece_conf_interval, brier_multi, brier_conf_interval
import wandb

sys.path.append('../../..')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TODO: define function for grl, compare ensemble of domain discriminator heads with classifier head post-hoc to baseline

def generate_conf_mat(pred: List[int], y_true: List[int], 
    class_labels: str, folder_path: str, title: str, 
    normalize: Optional[bool]=None, 
    figsize: Optional[Tuple[int]]=(15, 12)):
    conf_mat = confusion_matrix(y_true, pred, normalize=normalize)
    df_conf_mat = pd.DataFrame(conf_mat, index=class_labels, columns=class_labels)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df_conf_mat.to_csv(f'{folder_path}/{title}.csv')
    plt.figure(figsize=figsize)
    sn.heatmap(df_conf_mat, annot=True)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
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
        train_transform, val_transform, val_transform, val_transform
    ]

    datasets = CheckerboardOfficeHome(
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
    
    test_loader = DataLoader(datasets.test_dataset,
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
    if args.max_iters:
        num_backprop = args.max_iters
    else:
        num_backprop = args.iters_per_epoch * args.epochs
    # TODO: allow custom trade-off value for the gl
    if args.gl == 'warm':
        gl = WarmStartGradientLayer(alpha=args.alpha, lo=1., hi=args.lambda_c, max_iters=num_backprop, auto_step=True)
    elif args.gl == 'cool':
        gl = WarmStartGradientLayer(alpha=args.alpha, lo=0., hi=args.lambda_c, max_iters=num_backprop, auto_step=True)
    else:
        gl = GradientLayer(coeff=args.lambda_c)
    
    classifier = ImageClassifier(backbone,
                                 len(datasets.classes()),
                                 bottleneck_dim=args.bottleneck_dim,
                                 gl=gl).to(device)

    multidomain_discri = MultidomainDiscriminator(
        in_feature=classifier.features_dim,
        hidden_size=1024,
        num_domains=len(datasets.domains())).to(device)
    
    backbone_mul = 1.
    if args.finetune:
        backbone_mul = 0.1

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(backbone_lr=backbone_mul) +
                    multidomain_discri.get_parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(
        optimizer, lambda x: args.lr *
        (1. + args.lr_gamma * float(x))**(-args.lr_decay))

    
    # select the grl for the domain discriminator architecture
    if args.grl == 'warm':
        grl = WarmStartGradientReverseLayer(alpha=args.alpha, lo=0., hi=args.lambda_d, max_iters=num_backprop, auto_step=True)
    elif args.grl == 'cool':
        grl = WarmStartGradientReverseLayer(alpha=args.alpha, lo=1., hi=args.lambda_d, max_iters=num_backprop, auto_step=True)
    else:
        grl = GradientReverseLayer(coeff=args.lambda_d)
    multidomain_adv = MultidomainAdversarialLoss(multidomain_discri, grl=grl).to(device)

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
        train_domain_labels = CheckerboardOfficeHome.get_style(train_labels)
        test_features, test_labels = collect_feature_and_labels(
            test_loader, feature_extractor, device)
        test_domain_labels = CheckerboardOfficeHome.get_style(test_labels)
        val_features, val_labels = collect_feature_and_labels(
            val_loader, feature_extractor, device)
        val_domain_labels = CheckerboardOfficeHome.get_style(val_labels)
        novel_features, novel_labels = collect_feature_and_labels(
            novel_loader, feature_extractor, device)
        novel_domain_labels = CheckerboardOfficeHome.get_style(novel_labels)
        all_features = torch.cat([train_features, test_features, val_features, novel_features], axis=0)
        all_domain_labels = torch.cat([train_domain_labels, test_domain_labels, val_domain_labels, novel_domain_labels], axis=0)
        
        log = {}
        # plot t-SNE and calculate post-hoc domain discriminator accuracy
        log.update(domain_analyze(train_features, train_domain_labels, 'Train'))
        log.update(domain_analyze(test_features, test_domain_labels, 'Test'))
        log.update(domain_analyze(val_features, val_domain_labels, 'Validation'))
        log.update(domain_analyze(novel_features, novel_domain_labels, 'Novel'))
        log.update(domain_analyze(all_features, all_domain_labels, 'Total'))
        return log

    def partial_analysis(data_loader: DataLoader, dataset_type: str):
        feature_extractor = nn.Sequential(classifier.backbone,
                                        classifier.bottleneck).to(device)
        features, labels = collect_feature_and_labels(data_loader, 
                                        feature_extractor, device)
        domain_labels = CheckerboardOfficeHome.get_style(labels)
        return domain_analyze(features, domain_labels, dataset_type)  

    # analysis the model
    if args.phase == 'analysis':
        wandb.log(full_analysis(classifier))
        wandb.finish()
        return

    if args.phase == 'novel':
        acc1, novel_log, _ = validate(novel_loader, classifier, multidomain_adv,
                                   args, 'Novel', True)
        print(f'novel_acc1 = {acc1}')
        novel_log.update(partial_analysis(novel_log, 'Novel'))
        wandb.log(novel_log)
        wandb.finish()
        return

    # start training
    best_acc1 = 0.
    best_epoch = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train_log = train(train_iter, classifier, multidomain_adv, optimizer,
                          lr_scheduler, epoch, args)

        # evaluate on validation set
        # is_final_epoch = epoch == args.epochs - 1
        acc1, val_log, _ = validate(val_loader, classifier, multidomain_adv, args,
                                 'Validation',  gen_conf_mat=False, 
                                 calc_temp=False, gen_reli_diag=False, gen_rejection_curve=False)

        total_log = train_log.copy()
        total_log.update(val_log.copy())
        wandb.log(total_log)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path('latest'))
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path(f'epoch {epoch}'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path(f'epoch {epoch}'),
                        logger.get_checkpoint_path('best'))
            best_epoch = epoch
        best_acc1 = max(acc1, best_acc1)

    # load the model used for evaluation
    if args.use_best_model:
        classifier.load_state_dict(
            torch.load(logger.get_checkpoint_path('best')))
    else:
        classifier.load_state_dict(
            torch.load(logger.get_checkpoint_path('latest')))

    # evaluate best model on validation set with more information and temp calculation
    acc1, best_val_log, t = validate(val_loader, classifier, multidomain_adv, args,
                                 'Best Model on Validation',  gen_conf_mat=True, 
                                 conf_interval=True, calc_temp=True, gen_reli_diag=True)
    print("best_val_acc1 = {:3.1f}".format(acc1))
    
    # evaluate best model on validation set with more information and temp calculation
    acc1, test_log, _ = validate(test_loader, classifier, multidomain_adv, args,
                                 'Test',  gen_conf_mat=True, calc_temp=False, 
                                 conf_interval=True, input_temperature=t, gen_reli_diag=True)
    print("test_acc1 = {:3.1f}".format(acc1))

    # evaluate on novel set
    acc1, novel_log, _ = validate(novel_loader, classifier, multidomain_adv, args,
                               'Novel', gen_conf_mat=True, calc_temp=False, 
                               conf_interval=True, input_temperature=t, gen_reli_diag=True)
    print("novel_acc1 = {:3.1f}".format(acc1))
    
    eval_log = {"Best Category Classification Accuracy (Validation Set)": best_acc1,
                "Epoch of Best Validation Accuracy (Validation Set)": best_epoch}
    
    eval_log.update(best_val_log.copy())
    eval_log.update(test_log.copy())
    eval_log.update(novel_log.copy())
    eval_log.update(full_analysis(classifier))
    wandb.log(eval_log)

    wandb.finish()
    logger.close()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
# def get_grad_list(backbone: nn.Module):
#     return [param.grad.detach().clone() for name, param in backbone.named_parameters() if param.grad != None]

def train(train_iter: ForeverDataIterator, 
          model: ImageClassifier,
          multidomain_adv: MultidomainAdversarialLoss, 
          optimizer: SGD,
          lr_scheduler: LambdaLR,
          epoch: int, 
          args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    cls_losses = AverageMeter('Cls Loss', ':6.2f')
    transfer_losses = AverageMeter('Transfer Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, transfer_losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    multidomain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_tr, labels_tr = next(train_iter)

        # retrieve the class and domain from the checkerboard office_home dataset
        class_labels_tr = CheckerboardOfficeHome.get_category(labels_tr)
        domain_labels_tr = CheckerboardOfficeHome.get_style(labels_tr)

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
        total_loss = cls_loss + transfer_loss
        
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

    log = {}
        
    log.update({
        "Category Classification Loss (Training Set)": cls_losses.avg,
        'Style Discrimination Loss (Training Set)': transfer_losses.avg,
        # 'Total Loss (Training Set)': total_losses.sum,
        'Category Classification Accuracy (Training Set)': cls_accs.avg,
        'Style Discrimination Accuracy (Training Set)': domain_accs.avg,
        'Classification Logits': y_tr,
    })
    
    return log

def reliability_diag(conf: List[int], est_acc: List[int], density: List[int],
                      scaled_conf: List[int], scaled_est_acc: List[int], scaled_density: List[int],
                      log_path: str, dataset_type: str, figsize: Optional[Tuple[int]]=(7,7)):
    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], color='blue', label='Perfect Calibration')
    plt.scatter(conf, est_acc, density, color='red', label='Non-Calibrated')
    plt.scatter(scaled_conf, scaled_est_acc, scaled_density, color='green', label='Calibrated')
    title = f"Reliability Diagram ({dataset_type} Set)"
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Expected Accuracy")
    plt.legend()
    
    graph_folder_path = f'{log_path}/reliability-diagram/graph/'
    data_folder_path = f'{log_path}/reliability-diagram/data/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(graph_folder_path):
        os.makedirs(graph_folder_path)
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
        
    plt.savefig(f'{graph_folder_path}/{title}.png')
    np.save(f'{data_folder_path}/{title}_est_acc.npy', est_acc)
    np.save(f'{data_folder_path}/{title}_conf.npy', conf)
    np.save(f'{data_folder_path}/{title}_density.npy', density)
    np.save(f'{data_folder_path}/{title}_scaled_est_acc.npy', scaled_est_acc)
    np.save(f'{data_folder_path}/{title}_scaled_conf.npy', scaled_conf)
    np.save(f'{data_folder_path}/{title}_scaled_density.npy', scaled_density)


def calibration_evaluation(class_probs: List[List[int]], 
                           class_labels: List[int], 
                           classes: List[int],
                           dataset_type: str,
                           temp_scaled: Optional[bool] = False,
                           conf_interval: Optional[bool] = False,
                           confidence: Optional[float] = 0.95,
                           bootstrap_size: Optional[int] = 1000):
    log = {}
    ece, est_acc, conf, density = kernel_ece(class_probs, class_labels, classes, give_kde_points=True)
    
    add_on = ''
    if temp_scaled:
        add_on = 'Post-Temperature-Scaling ' 
    
    if conf_interval:
        ece, ece_lower, ece_upper = kernel_ece_conf_interval(class_probs, class_labels, classes, 
                                                     confidence=confidence, size=bootstrap_size)
        brier, brier_lower, brier_upper = brier_conf_interval(class_probs, class_labels, len(classes), confidence, bootstrap_size)
        log.update({f'KDE Expected Calibration Error Lower Bound {add_on}({dataset_type}, Confidence = {confidence})': ece_lower,
                    f'KDE Expected Calibration Error Upper Bound {add_on}({dataset_type}, Confidence = {confidence})': ece_upper,
                    f'Brier Score Lower Bound {add_on}({dataset_type}, Confidence = {confidence})': brier_lower,
                    f'Brier Score Upper Bound {add_on}({dataset_type}, Confidence = {confidence})': brier_upper})
    else:
        brier = brier_multi(class_probs, class_labels, len(classes))    
    


    print(f"KDE Expected Calibration Error {add_on}({dataset_type} Set): {ece}")
    print(f"Brier Score {add_on}({dataset_type} Set): {brier}")
    
    log.update({
        f'KDE Expected Calibration Error {add_on}({dataset_type})': ece,
        f'Brier Score {add_on}({dataset_type})': brier
    })

    return log, est_acc, conf, density

def temp_scale_probs(logits: torch.Tensor, temperature: int):
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=1).tolist() 
    
def rejection_data(probs: List[List[int]], labels: List[int], classes: List[int], use_fraction_rejected: Optional[bool]=False):
    assert len(probs) == len(labels)
    data = [(max(probs[i]), float(max(classes, key=lambda j: probs[i][j]) == labels[i])) for i in range(len(labels))]
    dtype = [('max_prob', float), ('acc', float)]
    data = np.array(data, dtype=dtype)
    data = np.sort(data, order='max_prob')
    total = 0.
    for i in range(1, data.size):
        total += data[data.size - 2 - i][1]
        data[data.size - 1 - i][1] = total
    for i in range(data.size):
        if use_fraction_rejected:
            data[i][0] = (i + 1)/data.size
        data[i][1] /= data.size - i
    return data

def rejection_curve(probs: List[List[int]], labels: List[int], classes: List[int],
                     log_path: str, dataset_type: str, temp_scaled: Optional[bool]=False, 
                     use_fraction_rejected: Optional[bool]=False, figsize: Optional[Tuple[int]]=(10,10)):
    data = rejection_data(probs, labels, classes, use_fraction_rejected=use_fraction_rejected)
    add_on = ''
    if temp_scaled:
        add_on = 'Post-Temperature-Scaling '
    
    plt.figure(figsize=figsize)
    plt.scatter(*zip(*data), s=1)
    
    if use_fraction_rejected:
        title = f'{add_on}Rejection Curve with Fraction Rejected ({dataset_type} Set)'
        plt.xlabel("Fraction Rejected")
    else:
        title = f'{add_on}Rejection Curve with Rejection Threshold ({dataset_type} Set)'
        plt.xlabel("Rejection Threshold")
    plt.ylabel("Accuracy")
    plt.title(title)
    
    graph_folder_path = f'{log_path}/rejection-curve/graph/{dataset_type}/'
    data_folder_path = f'{log_path}/rejection-curve/data/{dataset_type}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(graph_folder_path):
        os.makedirs(graph_folder_path)
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
    np.save(f'{data_folder_path}/{title}_data.npy', data)
    plt.savefig(f'{graph_folder_path}/{title}.png')
        


def validate(val_loader: DataLoader,
             model: ImageClassifier,
             multidomain_adv: MultidomainAdversarialLoss,
             args: argparse.Namespace,
             dataset_type: str,
             gen_conf_mat: Optional[bool] = False,
             calc_temp: Optional[bool] = False,
             input_temperature: Optional[int] = None,
             conf_interval: Optional[bool] = False,
             gen_reli_diag: Optional[bool] = False,
             gen_rejection_curve: Optional[bool] = False):
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
    log = {}

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            class_labels = CheckerboardOfficeHome.get_category(target)
            class_labels = class_labels.to(device)
            domain_labels = CheckerboardOfficeHome.get_style(target)
            domain_labels = domain_labels.to(device)

            # compute output
            class_pred, features = model(images)

            # compute loss
            cls_loss = F.cross_entropy(class_pred, class_labels)
            transfer_loss = multidomain_adv(features, domain_labels)

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
            # losses.update(loss.item(), images.size(0))

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
    classes = list(range(len(CheckerboardOfficeHome.CATEGORIES)))
    
    cal_log, est_acc, conf, density = calibration_evaluation(
                                        all_class_probs, 
                                        all_class_labels, 
                                        classes, 
                                        dataset_type,
                                        conf_interval=conf_interval
                                        # f'{args.log}/calibration/'
                                    )
    log.update(cal_log)
    calculated_temperature = None
    used_temperature = None

    if calc_temp:
        all_class_logits_list = all_class_logits.tolist()
        calculated_temperature = temp_scaling(all_class_logits_list, all_class_labels, len(classes))[0]
        print(f"Calculated Temperature: {calculated_temperature}")
        log.update({f"Calculated Temperature (Using {dataset_type} Set))": calculated_temperature})
        used_temperature = calculated_temperature
    elif input_temperature:
        used_temperature = input_temperature

    all_scaled_class_probs = None
    if used_temperature:
        all_scaled_class_probs = temp_scale_probs(all_class_logits, used_temperature)
        scaled_cal_log, scaled_est_acc, scaled_conf, scaled_density = calibration_evaluation(
                                                                                            all_scaled_class_probs, 
                                                                                            all_class_labels,
                                                                                            classes,
                                                                                            dataset_type,
                                                                                            conf_interval=conf_interval,
                                                                                            temp_scaled=True
                                                                                        )
        log.update(scaled_cal_log)
    
    if gen_reli_diag and used_temperature:
        reliability_diag(conf, est_acc, density, scaled_conf, 
                        scaled_est_acc, scaled_density,
                        args.log, dataset_type) 
    
    if gen_rejection_curve:
        # uncalibrated rejection curves
        rejection_curve(all_class_probs, all_class_labels, classes, args.log, dataset_type)
        rejection_curve(all_class_probs, all_class_labels, classes, args.log, dataset_type, use_fraction_rejected=True)
        if used_temperature: 
            # calibrated rejection curves
            rejection_curve(all_scaled_class_probs, all_class_labels, classes, args.log, dataset_type, temp_scaled=True)
            rejection_curve(all_scaled_class_probs, all_class_labels, classes, args.log, dataset_type, temp_scaled=True, use_fraction_rejected=True)

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
    
    log.update(
        {
            f"Category Classification Loss ({dataset_type} Set)": cls_losses.avg,
            f'Category Classification Accuracy ({dataset_type} Set)': top1.avg,
            f"Style Discriminator Loss ({dataset_type} Set)": transfer_losses.avg,
            f'Style Discriminator Accuracy ({dataset_type} Set)': domain_accs.avg,
            # f"Total Loss ({dataset_type} Set)": losses.sum
        }
    )
    return top1.avg, log, calculated_temperature


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
    # parser.add_argument('--lambda-trade-off',
    #                     default=0.5,
    #                     type=float,
    #                     help='the trade-off hyper-parameter between the classification loss and transfer loss')
    parser.add_argument('--lambda-c',
                        default=0.5,
                        type=float,
                        help='the trade-off hyper-parameter for classification head gl')
    parser.add_argument('--lambda-d',
                        default=0.5,
                        type=float,
                        help='the trade-off hyper-parameter for the domain discriminator head grl')
    parser.add_argument('--alpha',
                        default=1.,
                        type=float,
                        help='the alpha parameter in the warm and cool gradient layer')
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
    parser.add_argument('--max-iters',
                        '--warm-grl-max-iters',
                        default=None,
                        type=int,
                        help='Number of iterations for the warm grl/gl to reach max value')
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
    # parser.add_argument(
    #     '--largrange-multiplier',
    #     dest="use_lagrange_multiplier",
    #     default=False,
    #     action='store_true',
    #     help='''If true, make updates to lambda_trade_off based on the difference between 
    #     the loss for the domain discriminator and when the domain discriminator
    #     makes random guesses.''')
    # parser.add_argument(
    #     '--no-lagrange-multiplier',
    #     dest="use_best_model",
    #     default=False,
    #     action='store_false',
    #     help='''If false, make no updates to the lambda_trade_off.''')
    parser.add_argument(
        "--gl",
        type=str,
        default='warm',
        choices=['warm', 'cool', 'constant'],
        help="When gl is 'warm', the trade-off slowly increases from 0 to lambda."
        "When gl is 'cold', the trade-off slowly decreases from 1 to lambda."
        "When gl is 'constant', the trade-off is constant.")
    parser.add_argument(
        "--grl",
        type=str,
        default='warm',
        choices=['warm', 'cool', 'constant'],
        help="When grl is 'warm', the trade-off slowly increases from 0 to lambda."
        "When grl is 'cold', the trade-off slowly decreases from 1 to lambda."
        "When grl is 'constant', the trade-off is constant.")
    # parser.add_argument(
    #     '--warm-grl',
    #     dest="use_warm_grl",
    #     default=False,
    #     action='store_true',
    #     help='''If true, use the Warm Gradient Reversal Layer''')
    # parser.add_argument(
    #     '--no-warm-grl',
    #     dest="use_warm_grl",
    #     default=False,
    #     action='store_false',
    #     help='''If false, don't use the Warm Gradient Reversal Layer.''')
    # parser.add_argument(
    #     '--cool-grl',
    #     dest="use_cool_grl",
    #     default=False,
    #     action='store_true',
    #     help='''If true, use the Cool Gradient Reversal Layer''')
    # parser.add_argument(
    #     '--no-cool-grl',
    #     dest="use_cool_grl",
    #     default=False,
    #     action='store_false',
    #     help='''If false, don't use the Cool Gradient Reversal Layer.''')
    parser.add_argument(
        '--finetune',
        dest="finetune",
        default=True,
        action='store_true',
        help='''If true, set the learning rate of backbone to 0.1 of learning rate.''')
    parser.add_argument(
        '--no-finetune',
        dest="finetune",
        default=True,
        action='store_false',
        help='''If false, set the learning rate of backbone to 1.0 of learning rate.''')
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
