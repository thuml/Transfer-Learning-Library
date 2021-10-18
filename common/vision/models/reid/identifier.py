"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torch.nn import init


class ReIdentifier(nn.Module):
    r"""Person reIdentifier from `Bag of Tricks and A Strong Baseline for Deep Person Re-identification (CVPR 2019)
    <https://arxiv.org/pdf/1903.07071.pdf>`_.
    Given 2-d features :math:`f` from backbone network, the authors pass :math:`f` through another `BatchNorm1d` layer
    and get :math:`bn\_f`, which will then pass through a `Linear` layer to output predictions. During training, we
    use :math:`f` to compute triplet loss. While during testing, :math:`bn\_f` is used as feature. This may be a little
    confusing. The figures in the origin paper will help you understand better.
    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, finetune=True, pool_layer=None):
        super(ReIdentifier, self).__init__()
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            feature_bn = nn.BatchNorm1d(backbone.out_features)
            self.bottleneck = feature_bn
            self._features_dim = backbone.out_features
        else:
            feature_bn = nn.BatchNorm1d(bottleneck_dim)
            self.bottleneck = nn.Sequential(
                bottleneck,
                feature_bn
            )
            self._features_dim = bottleneck_dim

        self.head = nn.Linear(self.features_dim, num_classes, bias=False)
        self.finetune = finetune

        # initialize feature_bn and head
        feature_bn.bias.requires_grad_(False)
        init.constant_(feature_bn.weight, 1)
        init.constant_(feature_bn.bias, 0)
        init.normal_(self.head.weight, std=0.001)

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor):
        """"""
        f = self.pool_layer(self.backbone(x))
        bn_f = self.bottleneck(f)
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
