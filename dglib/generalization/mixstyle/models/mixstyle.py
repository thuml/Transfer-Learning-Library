"""
Modified from https://github.com/KaiyangZhou/mixstyle-release
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    r"""MixStyle module from `DOMAIN GENERALIZATION WITH MIXSTYLE (ICLR 2021) <https://arxiv.org/pdf/2104.02008v1.pdf>`_.
    Given input :math:`x`, we first compute mean :math:`\mu(x)` and standard deviation :math:`\sigma(x)` across spatial
    dimension. Then we permute :math:`x` and get :math:`\tilde{x}`, corresponding mean :math:`\mu(\tilde{x})` and
    standard deviation :math:`\sigma(\tilde{x})`. `MixUp` is performed using mean and standard deviation

    .. math::
        \gamma_{mix} = \lambda\sigma(x) + (1-\lambda)\sigma(\tilde{x})

    .. math::
        \beta_{mix} = \lambda\mu(x) + (1-\lambda)\mu(\tilde{x})

    where :math:`\lambda` is instance-wise weight sampled from `Beta distribution`. MixStyle is then

    .. math::
        MixStyle(x) = \gamma_{mix}\frac{x-\mu(x)}{\sigma(x)} + \beta_{mix}

    Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the `Beta distribution`.
          eps (float): scaling parameter to avoid numerical issues.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        batch_size = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sigma = (var + self.eps).sqrt()
        mu, sigma = mu.detach(), sigma.detach()
        x_normed = (x - mu) / sigma

        interpolation = self.beta.sample((batch_size, 1, 1, 1))
        interpolation = interpolation.to(x.device)

        # split into two halves and swap the order
        perm = torch.arange(batch_size - 1, -1, -1)  # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(batch_size // 2)]
        perm_a = perm_a[torch.randperm(batch_size // 2)]
        perm = torch.cat([perm_b, perm_a], 0)

        mu_perm, sigma_perm = mu[perm], sigma[perm]
        mu_mix = mu * interpolation + mu_perm * (1 - interpolation)
        sigma_mix = sigma * interpolation + sigma_perm * (1 - interpolation)

        return x_normed * sigma_mix + mu_mix
