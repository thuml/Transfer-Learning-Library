"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
from typing import Tuple, Optional, List, Dict
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import tqdm
from .lwf import Classifier as ClassifierBase

__all__ = ['Classifier', 'CoTuningLoss', 'Relationship']


class CoTuningLoss(nn.Module):
    """
    The Co-Tuning loss in `Co-Tuning for Transfer Learning (NIPS 2020)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/co-tuning-for-transfer-learning-nips20.pdf>`_.

    Inputs:
        - input: p(y_s) predicted by source classifier.
        - target: p(y_s|y_t), where y_t is the ground truth class label in target dataset.

    Shape:
        - input:  (b, N_p), where b is the batch size and N_p is the number of classes in source dataset
        - target: (b, N_p), where b is the batch size and N_p is the number of classes in source dataset
        - Outputs: scalar.
    """

    def __init__(self):
        super(CoTuningLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = - target * F.log_softmax(input, dim=-1)
        y = torch.mean(torch.sum(y, dim=-1))
        return y


class Relationship(object):
    """Learns the category relationship p(y_s|y_t) between source dataset and target dataset.

    Args:
        data_loader (torch.utils.data.DataLoader): A data loader of target dataset.
        classifier (torch.nn.Module): A classifier for Co-Tuning.
        device (torch.nn.Module): The device to run classifier.
        cache (str, optional): Path to find and save the relationship file.

    """
    def __init__(self, data_loader, classifier, device, cache=None):
        super(Relationship, self).__init__()
        self.data_loader = data_loader
        self.classifier = classifier
        self.device = device
        if cache is None or not os.path.exists(cache):
            source_predictions, target_labels = self.collect_labels()
            self.relationship = self.get_category_relationship(source_predictions, target_labels)
            if cache is not None:
                np.save(cache, self.relationship)
        else:
            self.relationship = np.load(cache)

    def __getitem__(self, category):
        return self.relationship[category]

    def collect_labels(self):
        """
        Collects predictions of target dataset by source model and corresponding ground truth class labels.

        Returns:
            - source_probabilities, [N, N_p], where N_p is the number of classes in source dataset
            - target_labels, [N], where 0 <= each number < N_t, and N_t is the number of classes in target dataset
        """

        print("Collecting labels to calculate relationship")
        source_predictions = []
        target_labels = []

        self.classifier.eval()
        with torch.no_grad():
            for i, (x, label) in enumerate(tqdm.tqdm(self.data_loader)):
                x = x.to(self.device)
                y_s = self.classifier(x)

                source_predictions.append(F.softmax(y_s, dim=1).detach().cpu().numpy())
                target_labels.append(label)

        return np.concatenate(source_predictions, 0), np.concatenate(target_labels, 0)

    def get_category_relationship(self, source_probabilities, target_labels):
        """
        The direct approach of learning category relationship p(y_s | y_t).

        Args:
            source_probabilities (numpy.array): [N, N_p], where N_p is the number of classes in source dataset
            target_labels (numpy.array): [N], where 0 <= each number < N_t, and N_t is the number of classes in target dataset

        Returns:
            Conditional probability, [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
        """
        N_t = np.max(target_labels) + 1  # the number of target classes
        conditional = []
        for i in range(N_t):
            this_class = source_probabilities[target_labels == i]
            average = np.mean(this_class, axis=0, keepdims=True)
            conditional.append(average)
        return np.concatenate(conditional)


class Classifier(ClassifierBase):
    """A Classifier used in `Co-Tuning for Transfer Learning (NIPS 2020)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/co-tuning-for-transfer-learning-nips20.pdf>`_..

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data.
        num_classes (int): Number of classes.
        head_source (torch.nn.Module): Classifier head of source model.
        head_target (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True


    Inputs:
        - x (tensor): input data fed to backbone

    Outputs:
        - y_s: predictions of source classifier head
        - y_t: predictions of target classifier head

    Shape:
        - Inputs: (b, *) where b is the batch size and * means any number of additional dimensions
        - y_s: (b, N), where b is the batch size and N is the number of classes
        - y_t: (b, N), where b is the batch size and N is the number of classes

    """
    def __init__(self, backbone: nn.Module, num_classes: int,  head_source,  **kwargs):
        super(Classifier, self).__init__(backbone, num_classes, head_source, **kwargs)

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.head_source.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head_target.parameters(), "lr": 1.0 * base_lr},
        ]
        return params
