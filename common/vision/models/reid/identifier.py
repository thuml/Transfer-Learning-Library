from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torch.nn import init


class ReIdentifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, finetune=True):
        super(ReIdentifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim
        self.feature_bn = nn.BatchNorm1d(self.features_dim)
        self.head = nn.Linear(self.features_dim, num_classes, bias=False)
        self.finetune = finetune

        # initialize feature_bn and head
        self.feature_bn.bias.requires_grad_(False)
        init.constant_(self.feature_bn.weight, 1)
        init.constant_(self.feature_bn.bias, 0)
        init.normal_(self.head.weight, std=0.001)

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)
        f = self.pool(f)
        bn_f = self.bottleneck(f)
        bn_f = self.feature_bn(bn_f)
        if not self.training:
            return bn_f
        predictions = self.head(bn_f)
        return predictions, f

    def get_parameters(self, base_lr=1.0, rate=0.1) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": rate * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params
