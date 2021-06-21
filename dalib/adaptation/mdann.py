from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.grl import WarmStartGradientReverseLayer
from common.modules.classifier import Classifier as ClassifierBase
from common.utils.metric import accuracy

__all__ = ['MultidomainAdversarialLoss']


class MultidomainAdversarialLoss(nn.Module):
    """
    The a Modified version of Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Multidomain adversarial loss measures the domain discrepancy among multiple domains.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        # TODO: Add cross-entropy formula

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, D)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source(s), :math:`f^s`
        - f_t (tensor): feature representations on target(s), :math:`f^t`
        - l_s (tensor): labels for the domain(s) of the source
        - l_t (tensor): labels for the domain(s) of the target
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain(s).
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain(s).

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        # TODO: Edit for multisource and multidomain adaptation
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, multidomain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None):
        super(MultidomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.multidomain_discriminator = multidomain_discriminator
        self.loss = lambda input, target, weight: \
            F.cross_entropy(
                input, target, weight=weight, reduction=reduction
            )
        # self.bce = lambda input, target, weight: \
        #     F.binary_cross_entropy(
        #         input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f: torch.Tensor, d_labels: torch.Tensor, 
                w: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(f)
        self.domain_pred = self.multidomain_discriminator(f)
        self.domain_discriminator_accuracy = accuracy(self.domain_pred, d_labels)[0]
        if w is None:
            w = torch.ones((self.domain_pred.shape[-1], 1)).to(f.device)
        return self.loss(self.domain_pred, d_labels, w)
    
    # def forward(self, f_s: torch.Tensor, d_label_s: torch.Tensor, 
    #             f_t: Optional[torch.Tensor] = None, d_label_t: Optional[torch.Tensor] = None,
    #             w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     f = self.grl(torch.cat((f_s, f_t), dim=0))
    #     d = self.multidomain_discriminator(f)
    #     d_s, d_t = d.chunk(2, dim=0)
    #     self.domain_discriminator_accuracy = accuracy(
    #         d, torch.cat((d_label_s, d_label_t), dim=0))[0]
    #     # TODO: Get num classes
    #     if w_s is None:
    #         w_s = torch.ones((d.shape[-1], 1)).to(f_s.device)
    #     if w_t is None:
    #         w_t = torch.ones((d.shape[-1], 1)).to(f_s.device)
    #     return 0.5 * (self.loss(d_s, d_label_s, w_s) + self.loss(d_t, d_label_t, w_t))
    #     #return 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + self.bce(d_t, d_label_t, w_t.view_as(d_t)))



class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(
            backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
