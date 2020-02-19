import torch
import torch.nn as nn
from torch.autograd import Function

__all__ = ['DomainDiscriminator', 'DomainAdversarialLoss']


class DomainDiscriminator(nn.Module):
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
        self.grl = GradientReverseLayer.apply

    def forward(self, x, alpha):
        x = self.grl(x, alpha)
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10.}]


class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator, size_average=None, reduce=None, reduction='mean'):
        super(DomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, f_s, f_t, alpha=1.):
        d_s = self.domain_discriminator(f_s, alpha)
        d_label_s = torch.ones((f_s.size(0), 1)).cuda()
        d_t = self.domain_discriminator(f_t, alpha)
        d_label_t = torch.zeros((f_t.size(0), 1)).cuda()
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class GradientReverseLayer(Function):

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
