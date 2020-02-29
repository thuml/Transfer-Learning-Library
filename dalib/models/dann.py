import torch
import torch.nn as nn
from ._util import WarmStartGradientReverseLayer, binary_accuracy
from .classifier import Classifier as ClassifierBase

__all__ = ['DomainDiscriminator', 'DomainAdversarialLoss']


class DomainDiscriminator(nn.Module):
    """Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the features with input size (N, F) come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    :param in_feature: dimension of the input feature
    :type in_feature: int
    :param hidden_size: dimension of the hidden features
    :type hidden_size: int

    Shape:
        - Input: :math:`(N, F)`
        - Output: :math:`(N, 1)`
    """
    def __init__(self, in_feature, hidden_size):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1.}]


class DomainAdversarialLoss(nn.Module):
    """The `Domain Adversarial Loss <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.

    ..
        TODO add DANN math definitions, explain what f_s, f_t means.

    :param domain_discriminator: A domain discriminator object, which predicts the domains
        of features. Its input shape is (N, F) and output shape is (N, 1)
    :type domain_discriminator: class:`nn.Module` object
    :param reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    :type reduction: string, optional

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:``reduction`` is ``'none'``, then :math:`(N, )`.

    Examples::
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> output = loss(f_s, f_t)
    """
    def __init__(self, domain_discriminator, reduction='mean'):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s, f_t):
        f_s = self.grl(f_s)
        d_s = self.domain_discriminator(f_s)
        d_label_s = torch.ones((f_s.size(0), 1)).cuda()

        f_t = self.grl(f_t)
        d_t = self.domain_discriminator(f_t)
        d_label_t = torch.zeros((f_t.size(0), 1)).cuda()

        self.grl.step()
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class Classifier(ClassifierBase):
    def __init__(self, backbone, num_classes, bottleneck_dim=256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(Classifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)
