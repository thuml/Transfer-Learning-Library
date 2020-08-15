from typing import Optional, List, Dict, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch

from dalib.modules.grl import WarmStartGradientReverseLayer


class MarginDisparityDiscrepancy(nn.Module):
    r"""The margin disparity discrepancy (MDD) is proposed to measure the distribution discrepancy in domain adaptation.

    The :math:`y^s` and :math:`y^t` are logits output by the main classifier on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial classifier.
    They are expected to contain raw, unnormalized scores for each class.

    The definition can be described as:

    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        \gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} \log\left(\frac{\exp(y_{adv}^s[h_{y^s}])}{\sum_j \exp(y_{adv}^s[j])}\right) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} \log\left(1-\frac{\exp(y_{adv}^t[h_{y^t}])}{\sum_j \exp(y_{adv}^t[j])}\right),

    where :math:`\gamma` is a margin hyper-parameter and :math:`h_y` refers to the predicted label when the logits output is :math:`y`.
    You can see more details in `Bridging Theory and Algorithm for Domain Adaptation <https://arxiv.org/abs/1904.05801>`_.

    Parameters:
        - **margin** (float): margin :math:`\gamma`. Default: 4
        - **reduction** (string, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs: y_s, y_s_adv, y_t, y_t_adv
        - **y_s**: logits output :math:`y^s` by the main classifier on the source domain
        - **y_s_adv**: logits output :math:`y^s` by the adversarial classifier on the source domain
        - **y_t**: logits output :math:`y^t` by the main classifier on the target domain
        - **y_t_adv**: logits output :math:`y_{adv}^t` by the adversarial classifier on the target domain

    Shape:
        - Inputs: :math:`(minibatch, C)` where C = number of classes, or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
          with :math:`K \geq 1` in the case of `K`-dimensional loss.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(minibatch)`, or
          :math:`(minibatch, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.

    Examples::
        >>> num_classes = 2
        >>> batch_size = 10
        >>> loss = MarginDisparityDiscrepancy(margin=4.)
        >>> # logits output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> # adversarial logits output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """

    def __init__(self, margin: Optional[int] = 4, reduction: Optional[str] = 'mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor) -> torch.Tensor:
        _, prediction_s = y_s.max(dim=1)
        _, prediction_t = y_t.max(dim=1)
        return self.margin * F.cross_entropy(y_s_adv, prediction_s, reduction=self.reduction) \
               + F.nll_loss(shift_log(1. - F.softmax(y_t_adv, dim=1)), prediction_t, reduction=self.reduction)


def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
    r"""
    First shift, then calculate log, which can be described as:

    .. math::
        y = \max(\log(x+\text{offset}), 0)

    Used to avoid the gradient explosion problem in log(x) function when x=0.

    Parameters:
        - **x**: input tensor
        - **offset**: offset size. Default: 1e-6

    .. note::
        Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
    """
    return torch.log(torch.clamp(x + offset, max=1.))


class ImageClassifier(nn.Module):
    r"""Classifier for MDD.
    Parameters:
        - **backbone** (class:`nn.Module` object): Any backbone to extract 1-d features from data
        - **num_classes** (int): Number of classes
        - **bottleneck_dim** (int, optional): Feature dimension of the bottleneck layer. Default: 1024
        - **width** (int, optional): Feature dimension of the classifier head. Default: 1024

    .. note::
        Classifier for MDD has one backbone, one bottleneck, while two classifier heads.
        The first classifier head is used for final predictions.
        The adversarial classifier head is only used when calculating MarginDisparityDiscrepancy.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

        >>> # x is inputs, classifier is an ImageClassifier
        >>> outputs, outputs_adv = classifier(x)
        >>> classifier.step()

    Inputs:
        - **x** (Tensor): input data

    Outputs: (outputs, outputs_adv)
        - **outputs**: logits outputs by the main classifier
        - **outputs_adv**: logits outputs by the adversarial classifier

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, C)`, where C means the number of classes.

    """

    def __init__(self, backbone: nn.Module, num_classes: int,
                 bottleneck_dim: Optional[int] = 1024, width: Optional[int] = 1024, finetune=True):
        super(ImageClassifier, self).__init__()
        self.backbone = backbone
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000., auto_step=False)

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
        self.finetune = finetune

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features = self.bottleneck(features)
        outputs = self.head(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.adv_head(features_adv)
        return outputs, outputs_adv

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GRL layer.
        """
        self.grl_layer.step()

    def get_parameters(self) -> List[Dict]:
        """
        :return: A parameters list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 if self.finetune else 1.},
            {"params": self.bottleneck.parameters(), "lr": 1.},
            {"params": self.head.parameters(), "lr": 1.},
            {"params": self.adv_head.parameters(), "lr": 1.}
        ]
        return params
