"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""
import random
import sys, os

import torch
import timm
from torch.utils.data import Subset
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models

sys.path.append('../../..')
import tllib.vision.datasets as datasets


class Logger(object):
    """Writes stream output to external text file.

    Args:
        filename (str): the file to write stream output
        stream: the stream to read from. Default: sys.stdout
    """

    def __init__(self, data_name, model_name, metric_name, stream=sys.stdout):
        self.terminal = stream
        self.save_dir = os.path.join(data_name, model_name)  # save intermediate features/outputs
        self.result_dir = os.path.join(data_name, f'{metric_name}.txt')  # save ranking results
        os.makedirs(self.save_dir, exist_ok=True)
        self.log = open(self.result_dir, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def get_save_dir(self):
        return self.save_dir

    def get_result_dir(self):
        return self.result_dir

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


def forwarding_dataset(score_loader, model, layer, device):
    """
    A forward forcasting on full dataset

    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params layer: before which layer features are extracted, for registering hooks
    
    returns
        features: extracted features of model
        prediction: probability outputs of model
        targets: ground-truth labels of dataset
    """
    features = []
    outputs = []
    targets = []

    def hook_fn_forward(module, input, output):
        features.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())

    forward_hook = layer.register_forward_hook(hook_fn_forward)

    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.to(device)
            _ = model(data)

    forward_hook.remove()

    features = torch.cat([x for x in features]).numpy()
    outputs = torch.cat([x for x in outputs])
    predictions = F.softmax(outputs, dim=-1).numpy()
    targets = torch.cat([x for x in targets]).numpy()

    return features, predictions, targets


def get_model(model_name, pretrained=True, pretrained_checkpoint=None):
    if model_name in get_model_names():
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrained)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrained)
    if pretrained_checkpoint:
        print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
        pretrained_dict = torch.load(pretrained_checkpoint)
        backbone.load_state_dict(pretrained_dict, strict=False)
    return backbone


def get_dataset(dataset_name, root, transform, sample_rate=100, num_samples_per_classes=None, split='train'):
    """
    When sample_rate < 100,  e.g. sample_rate = 50, use 50% data to train the model.
    Otherwise,
        if num_samples_per_classes is not None, e.g. 5, then sample 5 images for each class, and use them to train the model;
        otherwise, keep all the data.
    """
    dataset = datasets.__dict__[dataset_name]
    if sample_rate < 100:
        score_dataset = dataset(root=root, split=split, sample_rate=sample_rate, download=True, transform=transform)
        num_classes = len(score_dataset.classes)
    else:
        score_dataset = dataset(root=root, split=split, download=True, transform=transform)
        num_classes = len(score_dataset.classes)
        if num_samples_per_classes is not None:
            samples = list(range(len(score_dataset)))
            random.shuffle(samples)
            samples_len = min(num_samples_per_classes * num_classes, len(score_dataset))
            print("Origin dataset:", len(score_dataset), "Sampled dataset:", samples_len, "Ratio:",
                  float(samples_len) / len(score_dataset))
            dataset = Subset(score_dataset, samples[:samples_len])
    return score_dataset, num_classes


def get_transform(resizing='res.'):
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
    elif resizing == 'res.299':
        transform = T.Resize((299, 299))
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
