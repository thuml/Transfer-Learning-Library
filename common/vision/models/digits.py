import torch.nn as nn

class LeNet:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.bottleneck_dim = 50 * 4 * 4

    def backbone(self):
        return nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

    def bottleneck(self):
        return nn.Flatten(start_dim=1)

    def head(self):
        return nn.Sequential(
            nn.Linear(self.bottleneck_dim, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, self.num_classes)
        )

    def complete(self):
        return nn.Sequential(
            self.backbone(),
            self.bottleneck(),
            self.head()
        )


class DTN:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.bottleneck_dim = 256 * 4 * 4

    def backbone(self):
        return nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU(),
        )

    def bottleneck(self):
        return nn.Flatten(start_dim=1)

    def head(self):
        return nn.Sequential(
                nn.Linear(self.bottleneck_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, self.num_classes)
        )

    def complete(self):
        return nn.Sequential(
            self.backbone(),
            self.bottleneck(),
            self.head()
        )


def lenet(**kwargs):
    """LeNet model from
    `"Gradient-based learning applied to document recognition" <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_

    Args:
        num_classes (int): number of classes. Default: 10

    .. note::
        The input image size must be 28 x 28.

    Examples::
        >>> # Get the whole LeNet model
        >>> model = lenet().complete()
        >>> # Or combine it by yourself
        >>> model = nn.Sequential(lenet().backbone(), lenet().bottleneck(), lenet().head())
    """
    return LeNet(**kwargs)


def dtn(**kwargs):
    """ DTN model

    Args:
        num_classes (int): number of classes. Default: 10

    .. note::
        The input image size must be 32 x 32.

    Examples::
        >>> # Get the whole DTN model
        >>> model = dtn().complete()
        >>> # Or combine it by yourself
        >>> model = nn.Sequential(dtn().backbone(), dtn().bottleneck(), dtn().head())
    """
    return DTN(**kwargs)