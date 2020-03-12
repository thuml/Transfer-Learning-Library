import torch.nn as nn
import torch.nn.functional as F
import torch

from dalib.modules.grl import WarmStartGradientReverseLayer


class MarginDisparityDiscrepancy(nn.Module):
    r"""
    The margin disparity discrepancy (MDD) is proposed to measure the distribution discrepancy in domain adaptation.
    The definition can be described as:

    ..
        TODO add MDD math definitions, explain what y_s, y_s_adv, y_t, y_t_adv means.

    You can see more details in `Bridging Theory and Algorithm for Domain Adaptation`

    Parameters:
        - margin (float): margin gamma. Default: 2
        - reduction (string, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
          and :attr:`reduce` are in the process of being deprecated, and in the meantime,
          specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Output: scalar. If reduction is 'none', then `(N)`

    Examples::
        >>> num_classes = 2
        >>> batch_size = 10
        >>> loss = MarginDisparityDiscrepancy(margin=2.)
        >>> # logits output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> # adversarial logits output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """
    def __init__(self, margin=2, reduction='mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, y_s, y_s_adv, y_t, y_t_adv):
        _, prediction_s = y_s.max(dim=1)
        _, prediction_t = y_t.max(dim=1)
        return self.margin * F.cross_entropy(y_s_adv, prediction_s, reduction=self.reduction) \
               + F.nll_loss(shift_log(1.-F.softmax(y_t_adv, dim=1)), prediction_t, reduction=self.reduction)


def shift_log(input, offset=1e-6):
    r"""
    First shift, then calculate log.
    Used to avoid the gradient explosion problem in log(x) function when x=0.

    Parameters:
        - x: input tensor
        - offset:

    .. note::
        Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
    """
    return torch.log(torch.clamp(input + offset, max=1.))


class Classifier(nn.Module):
    r"""Classifier for MDD.
    Parameters:
        - backbone (class:`nn.Module` object): Any backbone to extract 1-d features from data
        - num_classes (int): Number of classes
        - bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024
        - width (int, optional): Feature dimension of the classifier head. Default: 1024

    .. note::
        Classifier for MDD has one backbone, one bottleneck, while two classifier heads.
        The first classifier head is used for final predictions.
        The adversarial classifier head is only used when calculating MarginDisparityDiscrepancy.
    """
    def __init__(self, backbone, num_classes, bottleneck_dim=1024, width=1024):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.grl_layer = WarmStartGradientReverseLayer()
        self.bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)
        # The classifier head used for final predictions.
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )
        # The adversarial classifier head
        self.adv_head = nn.Sequential(
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )
        for dep in range(2):
            self.head[dep * 3].weight.data.normal_(0, 0.01)
            self.head[dep * 3].bias.data.fill_(0.0)
            self.adv_head[dep * 3].weight.data.normal_(0, 0.01)
            self.adv_head[dep * 3].bias.data.fill_(0.0)

    def forward(self, inputs, keep_adv_output=False):
        """
        Parameters:
            - x (Tensor): input data
            - keep_adv_output (bool, optional)
            - return: Tuple (outputs, outputs_adv) if `keep_adv_output` is set True. Else only outputs.

        Shapes:
            - x: (N, *), same shape as the input of the `backbone`.
            - outputs, outputs_adv: (N, C), where C means the number of classes.
        """
        features = self.backbone(inputs)
        features = self.bottleneck(features)
        outputs = self.head(features)
        if keep_adv_output:
            features_adv = self.grl_layer(features)
            self.grl_layer.step()
            outputs_adv = self.adv_head(features_adv)
            return outputs, outputs_adv
        else:
            return outputs

    def get_parameters(self):
        """
        :return: A parameters list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
            {"params": self.adv_head.parameters(), "lr_mult": 1}
        ]
        return params

