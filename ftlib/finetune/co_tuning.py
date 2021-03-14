from typing import Tuple, Optional, List, Dict
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import tqdm

__all__ = ['Classifier', 'CoTuningLoss', 'Relationship']


class CoTuningLoss(nn.Module):
    def __init__(self):
        super(CoTuningLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = - target * F.log_softmax(input, dim=-1)
        y = torch.mean(torch.sum(y, dim=-1))
        return y


class Relationship(object):
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
        print("Collecting labels to calculate relationship")
        source_predictions = []
        target_labels = []

        self.classifier.eval()
        with torch.no_grad():
            for i, (x, label) in enumerate(tqdm.tqdm(self.data_loader)):
                x = x.to(self.device)
                y_s, _ = self.classifier(x)

                source_predictions.append(F.softmax(y_s, dim=1).detach().cpu().numpy())
                target_labels.append(label)

        return np.concatenate(source_predictions, 0), np.concatenate(target_labels, 0)

    def get_category_relationship(self, source_probabilities, target_labels):
        """
        The direct approach of learning category relationship p(y_s | y_t).

        Args:
            source_probabilities: [N, N_p], where N_p is the number of classes in source dataset
            target_labels: [N], where 0 <= each number < N_t, and N_t is the number of classes in target dataset

        Returns:
            [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
        """
        N_t = np.max(target_labels) + 1  # the number of target classes
        conditional = []
        for i in range(N_t):
            this_class = source_probabilities[target_labels == i]
            average = np.mean(this_class, axis=0, keepdims=True)
            conditional.append(average)
        return np.concatenate(conditional)


class Classifier(nn.Module):
    """
    """

    def __init__(self, backbone: nn.Module, num_classes: int,  head_source,
                 head_target: Optional[nn.Module] = None, finetune=True):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self._features_dim = self.backbone.out_features
        self.head_source = head_source
        if head_target is None:
            self.head_target = nn.Linear(self._features_dim, num_classes)
        else:
            self.head_target = head_target
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor):
        """"""
        f = self.backbone(x)
        f = self.bottleneck(f)
        y_s = self.head_source(f)
        y_t = self.head_target(f)
        return y_s, y_t

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
