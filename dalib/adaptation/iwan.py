"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Optional, List, Dict
import torch
import torch.nn as nn

from common.modules.classifier import Classifier as ClassifierBase


class ImportanceWeightModule(object):
    r"""
    Calculating class weight based on the output of discriminator.
    Introduced by `Importance Weighted Adversarial Nets for Partial Domain Adaptation (CVPR 2018) <https://arxiv.org/abs/1803.09210>`_

    Args:
        discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features.
            Its input shape is :math:`(N, F)` and output shape is :math:`(N, 1)`
        partial_classes_index (list[int], optional): The index of partial classes. Note that this parameter is \
            just for debugging, since in real-world dataset, we have no access to the index of partial classes. \
            Default: None.

    Examples::

        >>> domain_discriminator = DomainDiscriminator(1024, 1024)
        >>> importance_weight_module = ImportanceWeightModule(domain_discriminator)
        >>> num_iterations = 10000
        >>> for _ in range(num_iterations):
        >>>     # feature from source domain
        >>>     f_s = torch.randn(32, 1024)
        >>>     # importance weights for source instance
        >>>     w_s = importance_weight_module.get_importance_weight(f_s)
    """

    def __init__(self, discriminator: nn.Module, partial_classes_index: Optional[List[int]] = None):
        self.discriminator = discriminator
        self.partial_classes_index = partial_classes_index

    def get_importance_weight(self, feature):
        """
        Get importance weights for each instance.

        Args:
            feature (tensor): feature from source domain, in shape :math:`(N, F)`

        Returns:
            instance weight in shape :math:`(N, 1)`
        """
        weight = 1. - self.discriminator(feature)
        weight = weight / weight.mean()
        weight = weight.detach()
        return weight

    def get_partial_classes_weight(self, weights: torch.Tensor, labels: torch.Tensor):
        """
        Get class weight averaged on the partial classes and non-partial classes respectively.

        Args:
            weights (tensor): instance weight in shape :math:`(N, 1)`
            labels (tensor): ground truth labels in shape :math:`(N, 1)`

        .. warning::
            This function is just for debugging, since in real-world dataset, we have no access to the index of \
            partial classes and this function will throw an error when `partial_classes_index` is None.
        """
        assert self.partial_classes_index is not None

        weights = weights.squeeze()
        is_partial = torch.Tensor([label in self.partial_classes_index for label in labels]).to(weights.device)
        if is_partial.sum() > 0:
            partial_classes_weight = (weights * is_partial).sum() / is_partial.sum()
        else:
            partial_classes_weight = torch.tensor(0)

        not_partial = 1. - is_partial
        if not_partial.sum() > 0:
            not_partial_classes_weight = (weights * not_partial).sum() / not_partial.sum()
        else:
            not_partial_classes_weight = torch.tensor(0)
        return partial_classes_weight, not_partial_classes_weight


class ImageClassifier(ClassifierBase):
    r"""The Image Classifier for `Importance Weighted Adversarial Nets for Partial Domain Adaptation <https://arxiv.org/abs/1803.09210>`_
    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
