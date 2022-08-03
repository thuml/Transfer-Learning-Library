"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch

__all__ = ['Regressor']


class Regressor(nn.Module):
    """A generic Regressor class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_factors (int): Number of factors
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use `nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        The learning rate of this regressor is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Regressor.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: regressor's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_factors`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_factors: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim=-1, head: Optional[nn.Module] = None, finetune=True):
        super(Regressor, self).__init__()
        self.backbone = backbone
        self.num_factors = num_factors
        if bottleneck is None:
            feature_dim = backbone.out_features
            self.bottleneck = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(feature_dim, feature_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
            self._features_dim = feature_dim
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Sequential(
                nn.Linear(self._features_dim, num_factors),
                nn.Sigmoid()
            )
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = self.bottleneck(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


