import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dann import GradientReverseLayer, binary_accuracy


__all__ = ['ConditionalDomainDiscriminator', 'ConditionalDomainAdversarialLoss']


class ConditionalDomainDiscriminator(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(ConditionalDomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        # x = self.dropout2(self.relu2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10.}]


class ConditionalDomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator, max_iters, num_classes=-1,
                 features_dim=-1, randomized=False, entropy_conditioning=False,
                 reduction='mean'):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = GradientReverseLayer()
        self.entropy_conditioning = entropy_conditioning

        if randomized:
            assert num_classes > 0 and features_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes)
        else:
            self.map = MultiLinearMap()

        self.max_iters = max_iters
        self.iter_num = 0
        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight, reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)

    def forward(self, features, logits, domain=1):
        if domain == 1:
            self.iter_num += 1
        alpha = np.float(2.0 * 1. / (1.0 + np.exp(-10. * self.iter_num / self.max_iters)) - 1.)

        f = features
        g = F.softmax(logits, dim=1).detach()
        h = self.grl(self.map(f, g), alpha)
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


