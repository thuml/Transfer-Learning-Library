import torch.nn as nn

__all__ = ['Baseline']


class Baseline(nn.Module):

    def __init__(self, backbone, num_classes=1000, use_bottleneck=True, bottleneck_dim=256):
        super().__init__()
        self.backbone = backbone
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(backbone.out_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, num_classes)
        else:
            self.fc = nn.Linear(backbone.out_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck:
            x = self.bottleneck(x)
        return self.fc(x)

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 1.},
            {"params": self.fc.parameters(), "lr_mult": 10.},
        ]
        if self.use_bottleneck:
            params += [{"params": self.bottleneck.parameters(), "lr_mult": 10.}]
        return params
