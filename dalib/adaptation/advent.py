"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Sequential):
    """
    Domain discriminator model from
    `ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation (CVPR 2019) <https://arxiv.org/abs/1811.12833>`_

    Distinguish pixel-by-pixel whether the input predictions come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        num_classes (int): num of classes in the predictions
        ndf (int): dimension of the hidden features

    Shape:
        - Inputs: :math:`(minibatch, C, H, W)` where :math:`C` is the number of classes
        - Outputs: :math:`(minibatch, 1, H, W)`
    """
    def __init__(self, num_classes, ndf=64):
        super(Discriminator, self).__init__(
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
        )


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return F.binary_cross_entropy_with_logits(y_pred, y_truth_tensor)


class DomainAdversarialEntropyLoss(nn.Module):
    r"""The `Domain Adversarial Entropy Loss <https://arxiv.org/abs/1811.12833>`_

    Minimizing entropy with adversarial learning through training a domain discriminator.

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts
          the domains of predictions. Its input shape is :math:`(minibatch, C, H, W)` and output shape is :math:`(minibatch, 1, H, W)`

    Inputs:
        - logits (tensor): logits output of segmentation model
        - domain_label (str, optional): whether the data comes from source or target.
          Choices: ['source', 'target']. Default: 'source'

    Shape:
        - logits: :math:`(minibatch, C, H, W)` where :math:`C` means the number of classes
        - Outputs: scalar.

    Examples::

        >>> B, C, H, W = 2, 19, 512, 512
        >>> discriminator = Discriminator(num_classes=C)
        >>> dann = DomainAdversarialEntropyLoss(discriminator)
        >>> # logits output on source domain and target domain
        >>> y_s, y_t = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
        >>> loss = 0.5 * (dann(y_s, "source") + dann(y_t, "target"))
    """
    def __init__(self, discriminator: nn.Module):
        super(DomainAdversarialEntropyLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, logits, domain_label='source'):
        """
        """
        assert domain_label in ['source', 'target']
        probability = F.softmax(logits, dim=1)
        entropy = prob_2_entropy(probability)
        domain_prediciton = self.discriminator(entropy)
        if domain_label == 'source':
            return bce_loss(domain_prediciton, 1)
        else:
            return bce_loss(domain_prediciton, 0)

    def train(self, mode=True):
        r"""Sets the discriminator in training mode. In the training mode,
        all the parameters in discriminator will be set requires_grad=True.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation mode (``False``). Default: ``True``.
        """
        self.discriminator.train(mode)
        for param in self.discriminator.parameters():
            param.requires_grad = mode
        return self

    def eval(self):
        r"""Sets the module in evaluation mode. In the training mode,
        all the parameters in discriminator will be set requires_grad=False.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.
        """
        return self.train(False)
