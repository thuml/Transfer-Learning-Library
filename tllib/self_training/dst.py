"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.modules.classifier import Classifier


class ImageClassifier(Classifier):
    r"""
    Classifier with non-linear pseudo head :math:`h_{\text{pseudo}}` and worst-case estimation head
    :math:`h_{\text{worst}}` from `Debiased Self-Training for Semi-Supervised Learning <https://arxiv.org/abs/2202.07136>`_.
    Both heads are directly connected to the feature extractor :math:`\psi`. We implement end-to-end adversarial
    training procedure between :math:`\psi` and :math:`h_{\text{worst}}` by introducing a gradient reverse layer.
    Note that both heads can be safely discarded during inference, and thus will introduce no inference cost.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer.
        width (int, optional): Hidden dimension of the non-linear pseudo head and worst-case estimation head.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - outputs: predictions of the main head :math:`h`
        - outputs_adv: predictions of the worst-case estimation head :math:`h_{\text{worst}}`
        - outputs_pseudo: predictions of the pseudo head :math:`h_{\text{pseudo}}`

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - outputs, outputs_adv, outputs_pseudo: (minibatch, `num_classes`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, width=2048, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        self.pseudo_head = nn.Sequential(
            nn.Linear(self.features_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, self.num_classes)
        )
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False)
        self.adv_head = nn.Sequential(
            nn.Linear(self.features_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, self.num_classes)
        )

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        f_adv = self.grl_layer(f)
        outputs_adv = self.adv_head(f_adv)
        outputs = self.head(f)
        outputs_pseudo = self.pseudo_head(f)
        if self.training:
            return outputs, outputs_adv, outputs_pseudo
        else:
            return outputs

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.pseudo_head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.adv_head.parameters(), "lr": 1.0 * base_lr}
        ]

        return params

    def step(self):
        self.grl_layer.step()


def shift_log(x, offset=1e-6):
    """
    First shift, then calculate log for numerical stability.
    """

    return torch.log(torch.clamp(x + offset, max=1.))


class WorstCaseEstimationLoss(nn.Module):
    r"""
    Worst-case Estimation loss from `Debiased Self-Training for Semi-Supervised Learning <https://arxiv.org/abs/2202.07136>`_
    that forces the worst possible head :math:`h_{\text{worst}}` to predict correctly on all labeled samples
    :math:`\mathcal{L}` while making as many mistakes as possible on unlabeled data :math:`\mathcal{U}`. In the
    classification task, it is defined as:

    .. math::
        loss(\mathcal{L}, \mathcal{U}) =
        \eta' \mathbb{E}_{y^l, y_{adv}^l \sim\hat{\mathcal{L}}} -\log\left(\frac{\exp(y_{adv}^l[h_{y^l}])}{\sum_j \exp(y_{adv}^l[j])}\right) +
        \mathbb{E}_{y^u, y_{adv}^u \sim\hat{\mathcal{U}}} -\log\left(1-\frac{\exp(y_{adv}^u[h_{y^u}])}{\sum_j \exp(y_{adv}^u[j])}\right),

    where :math:`y^l` and :math:`y^u` are logits output by the main head :math:`h` on labeled data and unlabeled data,
    respectively. :math:`y_{adv}^l` and :math:`y_{adv}^u` are logits output by the worst-case estimation
    head :math:`h_{\text{worst}}`. :math:`h_y` refers to the predicted label when the logits output is :math:`y`.

    Args:
        eta_prime (float): the trade-off hyper parameter :math:`\eta'`.

    Inputs:
        - y_l: logits output :math:`y^l` by the main head on labeled data
        - y_l_adv: logits output :math:`y^l_{adv}` by the worst-case estimation head on labeled data
        - y_u: logits output :math:`y^u` by the main head on unlabeled data
        - y_u_adv: logits output :math:`y^u_{adv}` by the worst-case estimation head on unlabeled data

    Shape:
        - Inputs: :math:`(minibatch, C)` where C denotes the number of classes.
        - Output: scalar.

    """

    def __init__(self, eta_prime):
        super(WorstCaseEstimationLoss, self).__init__()
        self.eta_prime = eta_prime

    def forward(self, y_l, y_l_adv, y_u, y_u_adv):
        _, prediction_l = y_l.max(dim=1)
        loss_l = self.eta_prime * F.cross_entropy(y_l_adv, prediction_l)

        _, prediction_u = y_u.max(dim=1)
        loss_u = F.nll_loss(shift_log(1. - F.softmax(y_u_adv, dim=1)), prediction_u)

        return loss_l + loss_u
