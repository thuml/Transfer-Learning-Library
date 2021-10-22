"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch


class SymmetricMSELoss(nn.Module):
    r"""
    The Symmetric MSE Loss of two predictions :math:`z, z'` can be described as:
    .. math::
        L = \dfrac{1}{C}\cdot \sum_{i=1}^{b}\sum_{c=1}^{C}[z_i(c) - z'_i(c)]^2
    where :math:`C` is the number of classes.
    Inputs:
        - input1 (tensor): The prediction of the input with a random data augmentation.
        - input2 (tensor): Another prediction of the input with a random data augmentation.
    Shape:
        - input1: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - input2: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - Output: scalar.
    """

    def __init__(self):
        super(SymmetricMSELoss, self).__init__()

    def forward(self, input1, input2):
        assert input1.size() == input2.size()
        num_classes = input1.size()[1]
        return torch.sum((input1 - input2) ** 2) / num_classes


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class MeanTeacher(nn.Module):
    r"""A class for the architecture from `Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results (ICLR 2017) <https://openreview.net/references/pdf?id=ry8u21rtl>`_.
    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head_1 (torch.nn.Module, optional): Any head layer. Use one fully-connected linear layer by default
        head_2 (torch.nn.Module, optional): Any head layer. Use one fully-connected linear layer by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True
    Inputs:
        - x (tensor): input data fed to `backbone`
    Outputs:
        - predictions_1 (tensor): predictions of head_1
        - predictions_2 (tensor): predictions of head_2
        - features (tensor): features after `bottleneck` layer and before `head` layer
    Shape:
        - Inputs: :math:`(minibatch, *)` where * means, any number of additional dimensions
        - predictions_1: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - predictions_2: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - features: :math:`(b, features\_dim)` where :math:`b` is the batch size.
    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head_1: Optional[nn.Module] = None,
                 head_2: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(MeanTeacher, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim
        if head_1 is not None:
            self.head_1 = head_1
        else:
            self.head_1 = nn.Linear(self._features_dim, num_classes)
        if head_2 is not None:
            self.head_2 = head_2
        else:
            self.head_2 = nn.Linear(self._features_dim, num_classes)
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions_1 = self.head_1(f)
        predictions_2 = self.head_2(f)
        return predictions_1, predictions_2, f

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head_1.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head_2.parameters(), "lr": 1.0 * base_lr}
        ]

        return params
