from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from dalib.modules.grl import WarmStartGradientReverseLayer
from dalib.modules.classifier import Classifier as ClassifierBase
from ._util import binary_accuracy

__all__ = ['DomainAdversarialLoss']


class DomainAdversarialLoss(nn.Module):
    r"""The `Domain Adversarial Loss <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) &= \mathbb{E}_{x_i^s \sim \mathcal{D}_s} log[D(f_i^s)] \\
        &+ \mathbb{E}_{x_j^t \sim \mathcal{D}_t} log[1-D(f_j^t)].\\

    Parameters:
        - **domain_discriminator** (class:`nn.Module` object): A domain discriminator object, which predicts
          the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        - **reduction** (string, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        - **grl** (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs: f_s, f_t
        - **f_s** (tensor): feature representations on source domain, :math:`f^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:``reduction`` is ``'none'``, then :math:`(N, )`.

    Examples::
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different adversarial loss weight to each sample(using w_s & w_t)
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional[WarmStartGradientReverseLayer] = None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        return 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + self.bce(d_t, d_label_t, w_t.view_as(d_t)))


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)
