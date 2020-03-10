import torch.nn as nn

__all__ = ['Classifier']


class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation in image classification.

    Parameters:
        - backbone (class:`nn.Module` object): Any backbone to extract 1-d features from data
        - num_classes (int): Number of classes
        - bottleneck (class:`nn.Module` object, optional): Any bottleneck layer. Use no bottleneck by default
        - bottleneck_dim (int, optional): Feature dimension the of bottleneck layer. Default: 256

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride `get_parameters`.
    """
    def __init__(self, backbone, num_classes, bottleneck=None, bottleneck_dim=-1):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self.fc = nn.Linear(backbone.out_features, num_classes)
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self.fc = nn.Linear(bottleneck_dim, num_classes)

    @property
    def features_dim(self):
        """The dimension of features before the final fully connected layer"""
        return self.fc.in_features

    def forward(self, x, keep_features=False):
        """
        Parameters:
            - x (Tensor): input data
            - keep_features (bool, optional)
            - return: Tuple (output, features) if `keep_features` is set True. Else only the logit output.

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
        :return: A parameters list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.fc.parameters(), "lr_mult": 1.},
        ]
        return params
