import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DANN']


class DANN(nn.Module):

    def __init__(self, backbone, num_classes=1000, use_bottleneck=True, bottleneck_dim=256,
                 training=True, discriminator_hidden_dim=1024):
        super(DANN, self).__init__()
        self.backbone = backbone
        self.use_bottleneck = use_bottleneck
        self.training = training

        if self.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(backbone.out_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim)
            )
            self.fc = nn.Linear(bottleneck_dim, num_classes)
        else:
            self.fc = nn.Linear(backbone.out_features, num_classes)

        if self.training:
            self.domain_discriminator = DomainDiscriminator(self.fc.in_features, discriminator_hidden_dim)

    def forward(self, x, keep_features=False):
        f = self.backbone(x)
        f = f.view(x.size(0), -1)
        if self.use_bottleneck:
            f = self.bottleneck(f)
        y = self.fc(f)
        if keep_features:
            return y, f
        else:
            return y

    def forward_loss(self, x_s, x_t, labels_s):
        y_s, f_s = self(x_s, keep_features=True)
        cls_loss = F.cross_entropy(y_s, labels_s)
        d_s = self.domain_discriminator(f_s)
        _, f_t = self(x_t, keep_features=True)
        d_t = self.domain_discriminator(f_t)
        transfer_loss = F.binary_cross_entropy(d_s, torch.ones((x_s.size(0), 1)).cuda()) + \
                        F.binary_cross_entropy(d_t, torch.zeros((x_t.size(0), 1)).cuda())
        return cls_loss + transfer_loss

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 1.},
            {"params": self.fc.parameters(), "lr_mult": 10.},
        ]
        if self.use_bottleneck:
            params += [{"params": self.bottleneck.parameters(), "lr_mult": 10.}]
        if self.training:
            params += [{"params": self.domain_discriminator.parameters(), "lr_mult": 10.}]
        return params


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

    def forward(self, x):
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y



