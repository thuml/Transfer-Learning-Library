import torch.nn as nn

__all__ = ['Classifier']


class Classifier(nn.Module):
    """Classifier.

    :param backbone: Any backbone to extract 1-d features from data
    :type backbone: class:`nn.Module` object
    :param num_classes: Number of classes
    :type num_classes: int
    :param use_bottleneck: If True, use bottleneck layer after backbone. Default: True
    :type use_bottleneck: bool, optional
    :param bottleneck_dim: Feature dimension the of bottleneck layer. Default: 256
    :type bottleneck_dim: int, optional

    .. note::
        This classifier is used in many domain adaptation algorithms, but it is not the
        core of those algorithms. You can implement your own classifier and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy.
    """
    def __init__(self, backbone, num_classes, use_bottleneck=True, bottleneck_dim=256):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.use_bottleneck = use_bottleneck
        self.num_classes = num_classes
        if self.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(backbone.out_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim)
            )
            self.fc = nn.Linear(bottleneck_dim, num_classes)
        else:
            self.bottleneck = nn.Identity()
            self.fc = nn.Linear(backbone.out_features, num_classes)

    @property
    def features_dim(self):
        """The dimension of features before the final fully connected layer"""
        return self.fc.in_features

    def forward(self, x, keep_features=False):
        """
        Args:
            - x: (Tensor) input data
            - keep_features: (bool, optional)
        :return: Tuple (output, features) if `keep_features` is set True. Else only the logit output.

        Shapes:
            - x: (N, *), same shape as the input of the `backbone`.
            - output: (N, C), where C means the number of classes.
            - features: (N, F), where F means the dimension of the features.
        """
        f = self.backbone(x)
        f = f.view(x.size(0), -1)
        f = self.bottleneck(f)
        output = self.fc(f)
        if keep_features:
            return output, f
        else:
            return output

    def get_parameters(self):
        """
        :return: A parameters list which decides the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.fc.parameters(), "lr_mult": 1.},
        ]
        return params
