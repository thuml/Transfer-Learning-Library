import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dalib.modules.grl import WarmStartGradientReverseLayer
from dalib.modules.classifier import Classifier as ClassifierBase
from ._util import binary_accuracy


__all__ = ['DomainDiscriminator', 'ConditionalDomainAdversarialLoss', 'ImageClassifier']


class ConditionalDomainAdversarialLoss(nn.Module):
    r"""The `Conditional Domain Adversarial Loss <https://arxiv.org/abs/1705.10667>`_

    Conditional Domain adversarial loss measures the domain discrepancy through training a domain discriminator in a
    conditional manner. Given domain discriminator :math:`D`, feature representation :math:`f` and
    classifier predictions :math:`g`, the definition of CDAN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) &= \mathbb{E}_{x_i^s \sim \mathcal{D}_s} log[D(T(f_i^s, g_i^s))] \\
        &+ \mathbb{E}_{x_j^t \sim \mathcal{D}_t} log[1-D(T(f_j^t, g_j^t))],\\

    where :math:`T` is a `multi linear map` or `randomized multi linear map` which convert two tensors to a single tensor.

    Parameters:
        - **domain_discriminator** (class:`nn.Module` object): A domain discriminator object, which predicts the domains of
          features. Its input shape is (N, F) and output shape is (N, 1)
        - **entropy_conditioning** (bool, optional): If True, use entropy-aware weight to reweight each training example.
          Default: False
        - **randomized** (bool, optional): If True, use `randomized multi linear map`. Else, use `multi linear map`.
          Default: False
        - **num_classes** (int, optional): Number of classes. Default: -1
        - **features_dim** (int, optional): Dimension of input features. Default: -1
        - **randomized_dim** (int, optional): Dimension of features after randomized. Default: 1024
        - **reduction** (string, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    .. note::
        You need to provide `num_classes`, `features_dim` and `randomized_dim` **only when** `randomized`
        is set True.

    Inputs: g_s, f_s, g_t, f_t
        - **g_s** (tensor): unnormalized classifier predictions on source domain, :math:`g^s`
        - **f_s** (tensor): feature representations on source domain, :math:`f^s`
        - **g_t** (tensor): unnormalized classifier predictions on target domain, :math:`g^t`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - g_s, g_t: :math:`(minibatch, C)` where C means the number of classes.
        - f_s, f_t: :math:`(minibatch, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:``reduction`` is ``'none'``, then :math:`(minibatch, )`.

    Examples::
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim, hidden_size=1024)
        >>> loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(g_s, f_s, g_t, f_t)
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

    def forward(self, g_s, f_s, g_t, f_t):
        trans_loss_s, domain_acc_s = self._single_domain_forward(g_s, f_s, domain=1)
        trans_loss_t, domain_acc_t = self._single_domain_forward(g_t, f_t, domain=0)
        self.grl.step()
        self.domain_discriminator_accuracy = 0.5 * (domain_acc_s + domain_acc_t)
        return 0.5 * (trans_loss_s + trans_loss_t)

    def _single_domain_forward(self, logits, features, domain=1):
        """Perform forward on a single domain.
        domain = 1 means source domain, domain = 0 means target domain
        """
        f = features
        g = F.softmax(logits, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)
        d_label = torch.ones((f.size(0), 1)).to(logits.device) * domain
        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size
        return self.bce(d, d_label, weight.view_as(d)), binary_accuracy(d, d_label)


class DomainDiscriminator(nn.Module):
    r"""Domain discriminator model. See class:`dalib.adaptation.dann.DomainDiscriminator` for details.
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


class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map

    Given two inputs :math:`f` and :math:`g`, the definition is

    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),

    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.

    Parameters:
        - **features_dim** (int): dimension of input :math:`f`
        - **num_classes** (int): dimension of input :math:`g`
        - **output_dim** (int, optional): dimension of output tensor. Default: 1024

    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """
    def __init__(self, features_dim, num_classes, output_dim=1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f, g):
        f = torch.mm(self.Rf, f)
        g = torch.mm(self.Rg, g)
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map

    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """
    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f, g):
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)


def entropy(predictions):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    return H.sum(dim=1)


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone, num_classes, bottleneck_dim=256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)


class SentenceClassifier(ClassifierBase):
    def __init__(self, backbone, num_classes, config, bottleneck_dim=256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(bottleneck_dim, num_classes)
        )
        super(SentenceClassifier, self).__init__(backbone, num_classes, bottleneck=bottleneck, head=head, bottleneck_dim=bottleneck_dim)

    def get_parameters(self):
        """
        :return: A parameters list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in self.backbone.named_parameters() if not any(nd in n for nd in no_decay)], "lr_mult": 0.1},
            {"params": [p for n, p in self.backbone.named_parameters() if any(nd in n for nd in no_decay)],
             "lr_mult": 0.1, "weight_decay": 0.0},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params

