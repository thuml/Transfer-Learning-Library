import torch
import torch.nn as nn
import torch.nn.functional as F
from ._util import WarmStartGradientReverseLayer

__all__ = ['Classifier', 'AdversarialClassifier',
           'MarginDisparityDiscrepancyLoss', 'MarginDisparityDiscrepancy']


class MarginDisparityDiscrepancyLoss(nn.Module):
    def __init__(self, adversarial_classifier, margin):
        super(MarginDisparityDiscrepancyLoss, self).__init__()
        self.adversarial_classifier = adversarial_classifier
        self.mdd = MarginDisparityDiscrepancy(margin)
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000)

    def forward(self, y_s, f_s, y_t, f_t):
        y_s_adv = self.adversarial_classifier(f_s)
        y_t_adv = self.adversarial_classifier(self.grl(f_t))
        self.grl.step()
        # y_s, y_t = y_s.detach(), y_t.detach() # TODO is detach() needed?
        return self.mdd(y_s, y_s_adv, y_t, y_t_adv)


class MarginDisparityDiscrepancy(nn.Module):
    """
    The margin disparity discrepancy (MDD) is proposed to measure the distribution discrepancy in domain adaptation.
    The definition can be described as:
        TODO add MDD math definitions, explain what y_s, y_s_adv, y_t, y_t_adv means.

    You can see more details in `Bridging Theory and Algorithm for Domain Adaptation`
    Args:
        margin (float): margin gamma.

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Output: scalar. If reduction is 'none', then `(N)`
    """
    def __init__(self, margin, reduction='mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, y_s, y_s_adv, y_t, y_t_adv):
        _, prediction_s = y_s.max(dim=1)
        _, prediction_t = y_t.max(dim=1)
        return self.margin * F.nll_loss(F.log_softmax(y_s_adv, dim=1), prediction_s, reduction=self.reduction) \
               + F.nll_loss(shift_log(1.-F.softmax(y_t_adv, dim=1)), prediction_t, reduction=self.reduction)


def shift_log(input, offset=1e-6):
    return torch.log(torch.clamp(input + offset, max=1.))


class ClassifierHead(nn.Module):
    def __init__(self, in_features, out_features, use_bottleneck=True, bottleneck_dim=1024):
        super(ClassifierHead, self).__init__()
        self.use_bottleneck = use_bottleneck
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(in_features, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            in_features = bottleneck_dim
            self.bottleneck[0].weight.data.normal_(0, 0.01)
            self.bottleneck[0].bias.data.fill_(0.0)
        else:
            self.bottleneck = nn.Identity()
        self.fc = nn.Linear(in_features, out_features)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.bottleneck(x)
        y = self.fc(x)
        return y


class Classifier(nn.Module):
    """Classifier for MDD. Similar as `nn.dalib.models.classifier.Classifier`"""
    def __init__(self, backbone, num_classes, use_bottleneck=True, bottleneck_dim=1024, head_bottleneck_dim=1024):
        super(Classifier, self).__init__()
        self.backbone = backbone
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(backbone.out_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            in_features = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()
            in_features = backbone.out_features
        self.head = ClassifierHead(in_features, num_classes, use_bottleneck=True, bottleneck_dim=head_bottleneck_dim)
        self.use_bottleneck = use_bottleneck
        self.num_classes = num_classes

    def forward(self, x, keep_features=False):
        features = self.backbone(x)
        features = features.view(x.size(0), -1)
        features = self.bottleneck(features)
        if keep_features:
            return self.head(features), features
        else:
            return self.head(features)

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params


class AdversarialClassifier(ClassifierHead):
    def __init__(self, in_features, num_classes, bottleneck_dim):
        super(AdversarialClassifier, self).__init__(in_features, num_classes,
                                                    use_bottleneck=True, bottleneck_dim=bottleneck_dim)

    def get_parameters(self):
        params = [{"params": self.parameters(), "lr_mult": 1.}]
        return params
