from typing import List, Dict, Optional
import torch.nn as nn

__all__ = ['MultidomainDiscriminator']


class MultidomainDiscriminator(nn.Sequential):
    r"""Multidomain discriminator model modified from the domain discriminator model in
    `"Domain-Adversarial Training of Neural Networks" (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish the domain of the input features based on its domain label in the dataset (E.g. "Art", "Clipart", "Product", "Real World" from the Office-Home dataset).

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        num_classes (int): number of domains to predict (N)
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, N)`
    """
    def __init__(self,
                 in_feature: int,
                 hidden_size: int,
                 num_domains: int = 1,
                 batch_norm=True):
        if batch_norm:
            super(MultidomainDiscriminator,
                  self).__init__(nn.Linear(in_feature, hidden_size),
                                 nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, num_domains))
                                #  ,nn.Sigmoid()) # removing sigmoid to allow for scaling
        else:
            super(MultidomainDiscriminator,
                  self).__init__(nn.Linear(in_feature, hidden_size),
                                 nn.ReLU(inplace=True), nn.Dropout(0.5),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(inplace=True), nn.Dropout(0.5),
                                 nn.Linear(hidden_size, num_domains))
                                #  ,nn.Sigmoid())

    def get_parameters(self, base_lr: Optional[int] = 1.) -> List[Dict]:
        return [{"params": self.parameters(), "lr": base_lr}]
