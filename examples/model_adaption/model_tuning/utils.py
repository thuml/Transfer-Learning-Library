"""
@author: Liu Yong
@contact: liuyong1095556447@gmail.com
"""
import time
from PIL import Image
import timm
import numpy as np
import random
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.transforms as T
from torch.optim import SGD, Adam

sys.path.append('../../../..')
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.transforms import Denormalize
from tllib.modules.aggregation import ResNet, Bottleneck


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrained_checkpoint=None):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=True)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
            backbone.copy_head = backbone.get_classifier
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
            backbone.copy_head = lambda x: x.head
    if pretrained_checkpoint:
        print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
        pretrained_dict = torch.load(pretrained_checkpoint)
        backbone.load_state_dict(pretrained_dict, strict=False)
    return backbone


def get_dataset(dataset_name, root, train_transform, val_transform, sample_rate=100, num_samples_per_classes=None):
    """
    When sample_rate < 100,  e.g. sample_rate = 50, use 50% data to train the model.
    Otherwise,
        if num_samples_per_classes is not None, e.g. 5, then sample 5 images for each class, and use them to train the model;
        otherwise, keep all the data.
    """
    dataset = datasets.__dict__[dataset_name]
    if sample_rate < 100:
        train_dataset = dataset(root=root, split='train', sample_rate=sample_rate, download=True, transform=train_transform)
        test_dataset = dataset(root=root, split='test', sample_rate=100, download=True, transform=val_transform)
        num_classes = train_dataset.num_classes
    else:
        train_dataset = dataset(root=root, split='train', download=True, transform=train_transform)
        test_dataset = dataset(root=root, split='test', download=True, transform=val_transform)
        num_classes = train_dataset.num_classes
        if num_samples_per_classes is not None:
            samples = list(range(len(train_dataset)))
            random.shuffle(samples)
            samples_len = min(num_samples_per_classes * num_classes, len(train_dataset))
            print("Origin dataset:", len(train_dataset), "Sampled dataset:", samples_len, "Ratio:", float(samples_len) / len(train_dataset))
            train_dataset = Subset(train_dataset, samples[:samples_len])
    return train_dataset, test_dataset, num_classes


def validate(val_loader, model, args, device, visualize=None) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                if visualize is not None:
                    visualize(images[0], "val_{}".format(i))

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False):
    """
    resizing mode:
        - default: take a random resized crop of size 224 with scale in [0.2, 1.];
        - res: resize the image to 224;
        - res.|crop: resize the image to 256 and take a random crop of size 224;
        - res.sma|crop: resize the image keeping its aspect ratio such that the
            smaller side is 256, then take a random crop of size 224;
        – inc.crop: “inception crop” from (Szegedy et al., 2015);
        – cif.crop: resize the image to 224, zero-pad it by 28 on each side, then take a random crop of size 224.
    """
    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.2, 1.))
    elif resizing == 'res.':
        transform = T.Resize((224, 224))
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224)
        ])
    elif resizing == "res.sma|crop":
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'inc.crop':
        transform = T.RandomResizedCrop(224)
    elif resizing == 'cif.crop':
        transform = T.Compose([
            T.Resize((224, 224)),
            T.Pad(28),
            T.RandomCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default'):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
        – res.|crop: resize the image such that the smaller side is of size 256 and
            then take a central crop of size 224.
    """
    if resizing == 'default':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = T.Resize((224, 224))
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_optimizer(optimizer_name, params, lr, wd, momentum):
    '''
    Args:
        optimizer_name:
            - SGD
            - Adam
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate
        weight_decay: weight decay
        momentum: momentum factor for SGD
    '''
    if optimizer_name == 'SGD':
        optimizer = SGD(params=params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    elif optimizer_name == 'Adam':
        optimizer = Adam(params=params, lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError(optimizer_name)
    return optimizer


def visualize(image, filename):
    """
    Args:
        image (tensor): 3 x H x W
        filename: filename of the saving image
    """
    image = image.detach().cpu()
    image = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    image = image.numpy().transpose((1, 2, 0)) * 255
    Image.fromarray(np.uint8(image)).save(filename)


def load_pretrained_pkl(name, model_path='pretrained_models'):
    pretrained_model_path = {
        'imageNet': 'resnet50-19c8e357.pth',
        'moco': 'moco_v1_200ep_pretrain.pth.tar',
        'maskrcnn': 'maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        'deeplab': 'deeplabv3_resnet50_coco-cd0a2569.pth',
        'keyPoint': 'keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
    }

    pkl = torch.load(os.path.join(model_path, pretrained_model_path[name]))
    state_dict = {}

    if name == 'imageNet':
        for k, v in pkl.items():
            if k.startswith("fc."):
                continue
            state_dict[k] = v
    elif name == 'moco':
        pkl = pkl['state_dict']
        state_dict = {}
        for k, v in pkl.items():
            if not k.startswith("module.encoder_q."):
                continue
            k = k.replace("module.encoder_q.", "")
            if k.startswith("fc."):
                continue
            state_dict[k] = v
    elif name == 'maskrcnn':
        for k, v in pkl.items():
            if not k.startswith("backbone.body."):
                continue
            k = k.replace("backbone.body.", "")
            state_dict[k] = v
    elif name == 'deeplab':
        for k, v in pkl.items():
            if not k.startswith("backbone."):
                continue
            k = k.replace("backbone.", "")
            state_dict[k] = v
    elif name == 'keyPoint':
        for k, v in pkl.items():
            if not k.startswith("backbone.body."):
                continue
            k = k.replace("backbone.body.", "")
            state_dict[k] = v

    return state_dict


def build_homogeneous_model(models, **kwargs):
    assert len(models)>=1

    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs) # ada-resent50
    state_dict_new = model.state_dict()

    state_dicts = []
    for name in models:
        state_dicts.append(load_pretrained_pkl(name))

    for name in state_dicts[0]:
        wt = state_dict_new[name]
        ws = []
        for state_dict in state_dicts:
            ws.append(state_dict[name])

        if wt.size() == ws[0].size():
            for w in ws:
                state_dict_new[name] += w
            state_dict_new[name] /= len(models)
        else:
            state_dict_new[name] = torch.stack(ws, dim=0)

        if 'mean' in name:
            state_dict_new[name] = ws[0] * 0.0
        elif 'var' in name:
            state_dict_new[name] = ws[0] * 0.0 + 1.0

    model.load_state_dict(state_dict_new)

    return model
