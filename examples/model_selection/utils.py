"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""
import time
from PIL import Image
import timm
import numpy as np
import random
import sys, os

import torch
from torch.utils.data import Subset
import torchvision.transforms as T
from torchvision import datasets

sys.path.append('../../..')
import tllib.vision.models as models
from tllib.vision.transforms import Denormalize


import os
import sys
import time

class Logger(object):
    """Writes stream output to external text file.

    Args:
        filename (str): the file to write stream output
        stream: the stream to read from. Default: sys.stdout
    """
    def __init__(self, data_name, model_name, metric_name, stream=sys.stdout):
        self.terminal = stream
        self.save_dir = os.path.join(data_name, model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log = open(os.path.join(data_name, f'{metric_name}.txt'), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def get_savedir(self):
        return self.save_dir

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrained_checkpoint=None):
    # load models from pytorch-image-models
    backbone = timm.create_model(model_name, pretrained=True)
    if pretrained_checkpoint:
        print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
        pretrained_dict = torch.load(pretrained_checkpoint)
        backbone.load_state_dict(pretrained_dict, strict=False)
    return backbone


def get_score_dataset(dataset_name, root, transform, sample_rate=100, num_samples_per_classes=None):
    """
    When sample_rate < 100,  e.g. sample_rate = 50, use 50% data to train the model.
    Otherwise,
        if num_samples_per_classes is not None, e.g. 5, then sample 5 images for each class, and use them to train the model;
        otherwise, keep all the data.
    """
    if sample_rate < 100:
        score_dataset = dataset(root=root, split='train', sample_rate=sample_rate, download=True, transform=transform)
        num_classes = len(score_dataset.classes)
    else:
        score_dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
        num_classes = len(score_dataset.classes)
        if num_samples_per_classes is not None:
            samples = list(range(len(score_dataset)))
            random.shuffle(samples)
            samples_len = min(num_samples_per_classes * num_classes, len(score_dataset))
            print("Origin dataset:", len(score_dataset), "Sampled dataset:", samples_len, "Ratio:", float(samples_len) / len(score_dataset))
            dataset = Subset(score_dataset, samples[:samples_len])
    return score_dataset, num_classes


def get_score_transform(resizing='default'):
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
