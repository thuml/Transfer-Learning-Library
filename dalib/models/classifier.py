import torch.nn as nn

__all__ = ['Classifier']


class Classifier(nn.Module):

    def __init__(self, backbone, num_classes=1000, use_bottleneck=True, bottleneck_dim=256):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(backbone.out_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim)
            )
            self.fc = nn.Linear(bottleneck_dim, num_classes)
        else:
            self.fc = nn.Linear(backbone.out_features, num_classes)

    @property
    def features_dim(self):
        return self.fc.in_features

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

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 1.},
            {"params": self.fc.parameters(), "lr_mult": 10.},
        ]
        if self.use_bottleneck:
            params += [{"params": self.bottleneck.parameters(), "lr_mult": 10.}]
        return params
