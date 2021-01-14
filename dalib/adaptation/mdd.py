from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch

from dalib.modules.grl import WarmStartGradientReverseLayer


class MarginDisparityDiscrepancy(nn.Module):
    r"""The margin disparity discrepancy (MDD) is proposed to measure the distribution discrepancy in domain adaptation.

    The :math:`y^s` and :math:`y^t` are logits output by the main head on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial head.

    The definition can be described as:

    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        -\gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} L_s (y^s, y_{adv}^s) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} L_t (y^t, y_{adv}^t),

    where :math:`\gamma` is a margin hyper-parameter, :math:`L_s` refers to the disparity function defined on the source domain
    and :math:`L_t` refers to the disparity function defined on the target domain.
    You can see more details in `Bridging Theory and Algorithm for Domain Adaptation <https://arxiv.org/abs/1904.05801>`_.

    Parameters:
        - **source_disparity** (Callable): The disparity function defined on the source domain, :math:`L_s`.
        - **target_disparity** (Callable): The disparity function defined on the target domain, :math:`L_t`.
        - **margin** (float): margin :math:`\gamma`. Default: 4
        - **reduction** (string, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs: y_s, y_s_adv, y_t, y_t_adv
        - **y_s**: output :math:`y^s` by the main head on the source domain
        - **y_s_adv**: output :math:`y^s` by the adversarial head on the source domain
        - **y_t**: output :math:`y^t` by the main head on the target domain
        - **y_t_adv**: output :math:`y_{adv}^t` by the adversarial head on the target domain

    Examples::
        >>> num_outputs = 2
        >>> batch_size = 10
        >>> loss = MarginDisparityDiscrepancy(margin=4., source_disparity=F.l1_loss, target_disparity=F.l1_loss)
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """

    def __init__(self, source_disparity: Callable, target_disparity: Callable,
                 margin: Optional[float] = 4, reduction: Optional[str] = 'mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.source_disparity = source_disparity
        self.target_disparity = target_disparity

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor) -> torch.Tensor:
        loss = -self.margin * self.source_disparity(y_s, y_s_adv) + self.target_disparity(y_t, y_t_adv)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class ClassificationMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):
    r"""The margin disparity discrepancy (MDD) that measures the distribution discrepancy in domain adaptation
    for classification.
    When margin is equal to 1, it's also called disparity discrepancy (DD).

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
        >>> loss = ClassificationMarginDisparityDiscrepancy(margin=4.)
        >>> # logits output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> # adversarial logits output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """
    def __init__(self, margin: Optional[float] = 4, **kwargs):
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return F.cross_entropy(y_adv, prediction, reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return -F.nll_loss(shift_log(1. - F.softmax(y_adv, dim=1)), prediction, reduction='none')

        super(ClassificationMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin, **kwargs)


class RegressionMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):
    r"""The margin disparity discrepancy (MDD) that measures the distribution discrepancy in domain adaptation
    for regression.

    The :math:`y^s` and :math:`y^t` are logits output by the main regressor on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial regressor.
    They are expected to contain normalized values for each factors.

    The definition can be described as:

    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        -\gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} L (y^s, y_{adv}^s) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} L (y^t, y_{adv}^t),

    where :math:`\gamma` is a margin hyper-parameter and :math:`L` refers to the disparity function defined on both domains.
    You can see more details in `Bridging Theory and Algorithm for Domain Adaptation <https://arxiv.org/abs/1904.05801>`_.

    Parameters:
        - **loss_function** (Callable): The disparity function defined on both domains, :math:`L`.
        - **margin** (float): margin :math:`\gamma`. Default: 4
        - **reduction** (string, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs: y_s, y_s_adv, y_t, y_t_adv
        - **y_s**: logits output :math:`y^s` by the main regressor on the source domain
        - **y_s_adv**: logits output :math:`y^s` by the adversarial regressor on the source domain
        - **y_t**: logits output :math:`y^t` by the main regressor on the target domain
        - **y_t_adv**: logits output :math:`y_{adv}^t` by the adversarial regressor on the target domain

    Shape:
        - Inputs: :math:`(minibatch, F)` where F = number of factors, or :math:`(minibatch, F, d_1, d_2, ..., d_K)`
          with :math:`K \geq 1` in the case of `K`-dimensional loss.
        - Output: scalar. The same size as the target: :math:`(minibatch)`, or
          :math:`(minibatch, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.

    Examples::
        >>> num_outputs = 2
        >>> batch_size = 10
        >>> loss = RegressionMarginDisparityDiscrepancy(margin=4., loss_function=F.l1_loss)
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """
    def __init__(self, margin: Optional[float] = 1, loss_function=F.mse_loss, **kwargs):
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            return loss_function(y_adv, y.detach(), reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            return loss_function(y_adv, y.detach(), reduction='none')

        super(RegressionMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin, **kwargs)


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
        self.num_classes = num_classes
        self.backbone = backbone
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000., auto_step=False)

        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.bottleneck[2].weight.data.normal_(0, 0.005)
        self.bottleneck[2].bias.data.fill_(0.1)

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


class ImageRegressor(nn.Module):
    r"""Regressor for MDD.
    Parameters:
        - **backbone** (class:`nn.Module` object): Any backbone to extract 1-d features from data
        - **num_factors** (int): Number of factors
        - **bottleneck_dim** (int, optional): Feature dimension of the bottleneck layer. Default: 1024
        - **width** (int, optional): Feature dimension of the classifier head. Default: 1024

    .. note::
        Regressor for MDD has one backbone, one bottleneck, while two regressor heads.
        The first regressor head is used for final predictions.
        The adversarial regressor head is only used when calculating MarginDisparityDiscrepancy.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

        >>> # x is inputs, regressor is an ImageRegressor
        >>> outputs, outputs_adv = regressor(x)
        >>> regressor.step()

    Inputs:
        - **x** (Tensor): input data

    Outputs: (outputs, outputs_adv)
        - **outputs**: outputs by the main regressor
        - **outputs_adv**: outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, F)`, where F means the number of factors.

    """
    def __init__(self, backbone: nn.Module, num_factors: int,
                 bottleneck_dim: Optional[int] = 1024, width: Optional[int] = 1024, finetune=True):
        super(ImageRegressor, self).__init__()
        self.backbone = backbone
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(backbone.out_features, bottleneck_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(),
        )

        # The regressor head used for final predictions.
        self.head = nn.Sequential(
            nn.Conv2d(bottleneck_dim, width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(width, num_factors),
            nn.Sigmoid()
        )
        for layer in self.head:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)
        # The adversarial regressor head
        self.adv_head = nn.Sequential(
            nn.Conv2d(bottleneck_dim, width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(width, num_factors),
            nn.Sigmoid()
        )
        for layer in self.adv_head:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)
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
