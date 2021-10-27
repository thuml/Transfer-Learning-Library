"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dalib.modules.gl import WarmStartGradientLayer
from common.utils.metric.keypoint_detection import get_max_preds


class FastPseudoLabelGenerator2d(nn.Module):
    def __init__(self, sigma=2):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, heatmap: torch.Tensor):
        heatmap = heatmap.detach()
        height, width = heatmap.shape[-2:]
        idx = heatmap.flatten(-2).argmax(dim=-1) # B, K
        pred_h, pred_w = idx.div(width, rounding_mode='floor'), idx.remainder(width) # B, K
        delta_h = torch.arange(height, device=heatmap.device) - pred_h.unsqueeze(-1) # B, K, H
        delta_w = torch.arange(width, device=heatmap.device) - pred_w.unsqueeze(-1) # B, K, W
        gaussian = (delta_h.square().unsqueeze(-1) + delta_w.square().unsqueeze(-2)).div(-2 * self.sigma * self.sigma).exp() # B, K, H, W
        ground_truth = F.threshold(gaussian, threshold=1e-2, value=0.)

        ground_false = (ground_truth.sum(dim=1, keepdim=True) - ground_truth).clamp(0., 1.)
        return ground_truth, ground_false


class PseudoLabelGenerator2d(nn.Module):
    """
    Generate ground truth heatmap and ground false heatmap from a prediction.

    Args:
        num_keypoints (int): Number of keypoints
        height (int): height of the heatmap. Default: 64
        width (int): width of the heatmap. Default: 64
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    Inputs:
        - y: predicted heatmap

    Outputs:
        - ground_truth: heatmap conforming to Gaussian distribution
        - ground_false: ground false heatmap

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - ground_truth: :math:`(minibatch, K, H, W)`
        - ground_false: :math:`(minibatch, K, H, W)`
    """
    def __init__(self, num_keypoints, height=64, width=64, sigma=2):
        super(PseudoLabelGenerator2d, self).__init__()
        self.height = height
        self.width = width
        self.sigma = sigma

        heatmaps = np.zeros((width, height, height, width), dtype=np.float32)

        tmp_size = sigma * 3
        for mu_x in range(width):
            for mu_y in range(height):
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], width)
                img_y = max(0, ul[1]), min(br[1], height)

                heatmaps[mu_x][mu_y][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        self.heatmaps = heatmaps
        self.false_matrix = 1. - np.eye(num_keypoints, dtype=np.float32)

    def forward(self, y):
        B, K, H, W = y.shape
        y = y.detach()
        preds, max_vals = get_max_preds(y.cpu().numpy())  # B x K x (x, y)
        preds = preds.reshape(-1, 2).astype(np.int)
        ground_truth = self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W).copy()

        ground_false = ground_truth.reshape(B, K, -1).transpose((0, 2, 1))
        ground_false = ground_false.dot(self.false_matrix).clip(max=1., min=0.).transpose((0, 2, 1)).reshape(B, K, H, W).copy()
        return torch.from_numpy(ground_truth).to(y.device), torch.from_numpy(ground_false).to(y.device)


class RegressionDisparity(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator2d): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator2d(num_keypoints)
        >>> from common.vision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator2d, criterion: nn.Module):
        super(RegressionDisparity, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator

    def forward(self, y, y_adv, weight=None, mode='min'):
        assert mode in ['min', 'max']
        ground_truth, ground_false = self.pseudo_label_generator(y.detach())
        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)


class PoseResNet2d(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shape:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet2d, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)

        if self.training:
            return y, y_adv
        else:
            return y

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()