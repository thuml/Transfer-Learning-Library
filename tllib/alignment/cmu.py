"""
@author: Jinghan Gao, Baixu Chen
@contact: getterk@163.com, cbx_99_hasta@outlook.com
"""
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
from tllib.modules.classifier import Classifier as ClassifierBase


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f


class Ensemble(nn.Module):

    def __init__(self, in_feature, num_classes):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(in_feature, num_classes)
        self.fc2 = nn.Linear(in_feature, num_classes)
        self.fc3 = nn.Linear(in_feature, num_classes)
        self.fc4 = nn.Linear(in_feature, num_classes)
        self.fc5 = nn.Linear(in_feature, num_classes)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc5.weight)

    def forward(self, x, index):
        if index == 0:
            y = self.fc1(x)
        elif index == 1:
            y = self.fc2(x)
        elif index == 2:
            y = self.fc3(x)
        elif index == 3:
            y = self.fc4(x)
        elif index == 4:
            y = self.fc5(x)
        else:
            y_1 = self.fc1(x)
            y_1 = nn.Softmax(dim=-1)(y_1)
            y_2 = self.fc2(x)
            y_2 = nn.Softmax(dim=-1)(y_2)
            y_3 = self.fc3(x)
            y_3 = nn.Softmax(dim=-1)(y_3)
            y_4 = self.fc4(x)
            y_4 = nn.Softmax(dim=-1)(y_4)
            y_5 = self.fc5(x)
            y_5 = nn.Softmax(dim=-1)(y_5)
            return y_1, y_2, y_3, y_4, y_5

        return y

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.parameters(), "lr": 1.0 * base_lr},
        ]
        return params


def get_marginal_confidence(y_1, y_2, y_3, y_4, y_5):
    conf_1, _ = torch.topk(y_1, 2, 1)
    conf_2, _ = torch.topk(y_2, 2, 1)
    conf_3, _ = torch.topk(y_3, 2, 1)
    conf_4, _ = torch.topk(y_4, 2, 1)
    conf_5, _ = torch.topk(y_5, 2, 1)

    conf_1 = conf_1[:, 0] - conf_1[:, 1]
    conf_2 = conf_2[:, 0] - conf_2[:, 1]
    conf_3 = conf_3[:, 0] - conf_3[:, 1]
    conf_4 = conf_4[:, 0] - conf_4[:, 1]
    conf_5 = conf_5[:, 0] - conf_5[:, 1]

    confidence = (conf_1 + conf_2 + conf_3 + conf_4 + conf_5) / 5
    return confidence


def get_entropy(y_1, y_2, y_3, y_4, y_5):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy2 = torch.sum(- y_2 * torch.log(y_2 + 1e-10), dim=1)
    entropy3 = torch.sum(- y_3 * torch.log(y_3 + 1e-10), dim=1)
    entropy4 = torch.sum(- y_4 * torch.log(y_4 + 1e-10), dim=1)
    entropy5 = torch.sum(- y_5 * torch.log(y_5 + 1e-10), dim=1)
    num_classes = y_1.size(1)
    entropy_norm = np.log(num_classes)

    entropy = (entropy1 + entropy2 + entropy3 + entropy4 + entropy5) / (5 * entropy_norm)
    return entropy


def norm(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x
