from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from dalib.modules.classifier import Classifier as ClassifierBase

__all__ = ['MinimumClassConfusionLoss', 'ImageClassifier']


class MinimumClassConfusionLoss(nn.Module):
    r"""The `Minimum Class Confusion Loss <https://arxiv.org/abs/1912.03699>`_

    Minimum Class Confusion loss minimizes the class confusion in the target predictions.
    Given classifier predictions (logits before softmax) :math:`Z`, the definition of MCC loss is

    .. math::
           {\widehat Y_{ij}} = \frac{{\exp \left( {{Z_{ij}}/T} \right)}}{{\sum\nolimits_{j' = 1}^{|{\mathcal{C}}|}
           {\exp \left( {{Z_{ij'}}/T} \right)} }},
           where :math:`T` is the temperature for rescaling,
           {{\mathbf{C}}_{jj'}} = {\widehat{\mathbf{y}}}_{ \cdot j}^{\sf T}{{\widehat{\mathbf{y}}}_{ \cdot j'}},
           H(\widehat{\bf y}_{i\cdot})= - { \sum _{j=1 }^{ |{\cal {C}}| }{ { \widehat { Y }  }_{ ij }\
           log{ \widehat { Y }  }_{ ij } }  },
           {W_{ii}} = \frac{{B\left( {1 + \exp ( { - H( {{{{\widehat{\bf y}}}_{i \cdot }}} )} )} \right)}}
           {{\sum\limits_{i' = 1}^B {\left( {1 + \exp ( { - H( {{{{\widehat{\bf y}}}_{i' \cdot }}} )} )} \right)} }},
           {{\mathbf{C}}_{jj'}} = {\widehat{\mathbf{y}}}_{ \cdot j}^{\sf T}{\mathbf{W}}{{\widehat{\mathbf{y}}}_{ \cdot j'}}.
           {{{\widetilde{\mathbf C}}}_{jj'}} = \frac{{{{\mathbf{C}}_{jj'}}}}{{\sum\nolimits_{{j''} = 1}^
           {|{\mathcal{C}}|} {{{\mathbf{C}}_{j{j''}}}} }},
           {L_{{\rm{MCC}}}} ( {{{\widehat {\mathbf{Y}}}_t}} ) = \frac{1}{|{\cal {C}}|}\sum\limits_{j = 1}^
           {|{\mathcal{C}}|} {\sum\limits_{j' \ne j}^{|{\mathcal{C}}|} {\left| {{{{\widetilde{\mathbf C}}}_{jj'}}} \right|} }.

    You can see more details in `Minimum Class Confusion for Versatile Domain Adaptation <https://arxiv.org/abs/1912.03699>`

    Parameters:
        - **temperature** (float) : The temperature for rescaling, the prediction will shrink to vanilla softmax if
          temperature is 1.0.

    .. note::
        Make sure that temperature is larger than 0.

    Inputs: g_t
        - **g_t** (tensor): unnormalized classifier predictions on target domain, :math:`g^t`

    Shape:
        - g_t: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.

    Examples::
        >>> temperature = 2.0
        >>> loss = MinimumClassConfusionLoss(temperature)
        >>> # logits output from target domain
        >>> g_t = torch.randn(batch_size, num_classes)
        >>> output = loss(g_t)

    MCC can also serve as a regularizer for existing methods.
    Examples::
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> temperature = 2.0
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim, hidden_size=1024)
        >>> cdan_loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> mcc_loss = MinimumClassConfusionLoss(temperature)
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> total_loss = cdan_loss(g_s, f_s, g_t, f_t) + mcc_loss(g_t)
    """

    def __init__(self, temperature):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, g_t: torch.Tensor) -> torch.Tensor:
        train_bs, class_num = g_t.size(0), g_t.size(1)
        g_t_temp = g_t / self.temperature
        g_t_temp_softmax = nn.Softmax(dim=1)(g_t_temp)
        target_entropy_weight = entropy(g_t_temp_softmax).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
        c_matrix = g_t_temp_softmax.mul(target_entropy_weight.view(-1,1)).transpose(1,0).mm(g_t_temp_softmax)
        c_matrix = c_matrix / torch.sum(c_matrix, dim=1)
        mcc_loss = (torch.sum(c_matrix) - torch.trace(c_matrix)) / class_num
        return mcc_loss


def entropy(predictions: torch.Tensor) -> torch.Tensor:
    r"""Entropy of N predictions :math:`(p_1, p_2, ..., p_N)`.
    The definition is:

    .. math::
        d(p_1, p_2, ..., p_N) = -\dfrac{1}{K} \sum_{k=1}^K \log \left( \dfrac{1}{N} \sum_{i=1}^N p_{ik} \right)

    where K is number of classes.

    Parameters:
        - **predictions** (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    return H.sum(dim=1)


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
