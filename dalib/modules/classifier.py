import torch.nn as nn

__all__ = ['Classifier']


class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation in image classification.

    Parameters:
        - **backbone** (class:`nn.Module` object): Any backbone to extract 1-d features from data
        - **num_classes** (int): Number of classes
        - **bottleneck** (class:`nn.Module` object, optional): Any bottleneck layer. Use no bottleneck by default
        - **bottleneck_dim** (int, optional): Feature dimension of the bottleneck layer. Default: 256
        - **head** (class:`nn.Module` object, optional): Any classifier head. Use `nn.Linear` by default

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride `get_parameters`.

    Inputs:
        - **x** (tensor): input data fed to `backbone`

    Outputs: predictions, features
        - **predictions**: classifier's predictions
        - **features**: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """
    def __init__(self, backbone, num_classes, bottleneck=None, bottleneck_dim=-1, head=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head

    @property
    def features_dim(self):
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x):
        """"""
        f = self.backbone(x)
        f = f.view(-1, self.backbone.out_features)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self):
        """A parameters list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params


class SentenceClassifier(Classifier):
    def __init__(self, backbone, num_classes, config):
        head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_classes)
        )
        super(SentenceClassifier, self).__init__(backbone, num_classes, bottleneck=None, head=head)

    def get_parameters(self):
        """A parameters list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in self.backbone.named_parameters() if not any(nd in n for nd in no_decay)], "lr_mult": 1.},
            {"params": [p for n, p in self.backbone.named_parameters() if any(nd in n for nd in no_decay)],
             "lr_mult": 1., "weight_decay": 0.0},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params

