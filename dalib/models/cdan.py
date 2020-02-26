import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._util import binary_accuracy, WarmStartGradientReverseLayer


__all__ = ['DomainDiscriminator', 'ConditionalDomainAdversarialLoss']


class DomainDiscriminator(nn.Module):
    """Domain discriminator model. See class:`dalib.models.dann.DomainDiscriminator` for details.
    """
    def __init__(self, in_feature, hidden_size):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        y = self.sigmoid(self.layer2(x))
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1.}]


class ConditionalDomainAdversarialLoss(nn.Module):
    """The `Conditional Domain Adversarial Loss <https://arxiv.org/abs/1705.10667>`_

    Conditional Domain adversarial loss measures the domain discrepancy through training a domain discriminator in a conditional manner.

    ..
        TODO add CDAN math definitions, explain what y_s, f_s, y_t, f_t means.

    :param domain_discriminator: A domain discriminator object, which predicts the domains
        of features. Its input shape is (N, F) and output shape is (N, 1)
    :type domain_discriminator: class:`nn.Module` object
    :param entropy_conditioning: If True, use entropy-aware weight to reweight each training
        example. Default: False
    :type entropy_conditioning: bool, optional
    :param randomized: If True, use randomized multi linear map. Else, use multi linear map.
        Default: False
    :type randomized: bool, optional
    :param num_classes: Number of classes
    :type num_classes: int, optional
    :param features_dim: Dimension of input features
    :type features_dim: int, optional
    :param randomized_dim: Dimension of features after randomized. Default: 1024
    :type randomized_dim: int, optional
    :param reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    :type reduction: string, optional

    .. note::
        You need to provide num_classes, features_dim and randomized_dim `only when` randomized
        is set True.

    Shape:
        - y_s, y_t: :math:`(N, C)` where C means the number of classes.
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:``reduction`` is ``'none'``, then :math:`(N, )`.

    Examples::
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim, hidden_size=1024)
        >>> loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(y_s, f_s, y_t, f_t)
    """
    def __init__(self, domain_discriminator, entropy_conditioning=False, randomized=False,
                 num_classes=-1, features_dim=-1, randomized_dim=1024, reduction='mean'):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000)
        self.entropy_conditioning = entropy_conditioning

        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()

        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight, reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, y_s, f_s, y_t, f_t):
        trans_loss_s, domain_acc_s = self._single_domain_forward(y_s, f_s, domain=1)
        trans_loss_t, domain_acc_t = self._single_domain_forward(y_t, f_t, domain=0)
        self.grl.step()
        self.domain_discriminator_accuracy = 0.5 * (domain_acc_s + domain_acc_t)
        return 0.5 * (trans_loss_s + trans_loss_t)

    def _single_domain_forward(self, logits, features, domain=1):
        f = features
        g = F.softmax(logits, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)
        d_label = torch.ones((f.size(0), 1)).cuda() * domain
        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size
        return self.bce(d, d_label, weight.view_as(d)), binary_accuracy(d, d_label)


class RandomizedMultiLinearMap(nn.Module):
    def __init__(self, features_dim, num_classes, output_dim=1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, features, predictions):
        features = torch.mm(self.Rf, features)
        predictions = torch.mm(self.Rg, predictions)
        output = torch.mul(features, predictions) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, features, predictions):
        batch_size = features.size(0)
        output = torch.bmm(predictions.unsqueeze(2), features.unsqueeze(1))
        return output.view(batch_size, -1)


def entropy(predictions):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    return H.sum(dim=1)


