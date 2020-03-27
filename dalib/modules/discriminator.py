import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    r"""Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Parameters:
        - **in_feature** (int): dimension of the input feature
        - **hidden_size** (int): dimension of the hidden features

    Shape:
        - Inputs: :math:`(N, F)`
        - Outputs: :math:`(N, 1)`
    """
    def __init__(self, in_feature, hidden_size, num_layers, use_dropout=False):
        super(DomainDiscriminator, self).__init__()

        layers = [self._make_layers(in_feature, hidden_size)]
        for i in range(1, num_layers-1):
            layers.append(self._make_layers(hidden_size, hidden_size, use_dropout))
        self.bottoleneck = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottoleneck(x)
        return self.head(x)

    def _make_layers(self, in_feature, out_features, use_dropout=False):
        if not use_dropout:
            return nn.Sequential(
                nn.Linear(in_feature, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Linear(in_feature, out_features),
                nn.ReLU(),
                nn.Dropout()
            )

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1.}]
