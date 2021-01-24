from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Sequential):
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
    def __init__(self, discriminator: nn.Module):
        super(DomainAdversarialEntropyLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, logits, domain_label='source'):
        assert domain_label in ['source', 'target']
        probability = F.softmax(logits, dim=1)
        entropy = prob_2_entropy(probability)
        domain_prediciton = self.discriminator(entropy)
        if domain_label == 'source':
            return bce_loss(domain_prediciton, 0)
        else:
            return bce_loss(domain_prediciton, 1)

    def train(self, mode=True):
        self.discriminator.train(mode)
        for param in self.discriminator.parameters():
            param.requires_grad = mode

    def eval(self):
        self.train(False)
