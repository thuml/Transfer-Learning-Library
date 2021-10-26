"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def update_bn(model, ema_model):
    for m2, m1 in zip(ema_model.named_modules(), model.named_modules()):
        if ('bn' in m2[0]) and ('bn' in m1[0]):
            bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
            bn2['running_mean'].data.copy_(bn1['running_mean'].data)
            bn2['running_var'].data.copy_(bn1['running_var'].data)
            bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)


class FixMatchConsistencyLoss(nn.Module):
    r"""The consistency loss from `FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (NIPS 2020) <https://proceedings.neurips.cc/paper/2020/file/f7ac67a9aa8d255282de7d11391e1b69-Paper.pdf>`_.
    It can be described as

    .. math::
        p^{weak}_i(c) = \dfrac{\exp (z^{weak}_i(c)/\tau)}{\sum_{k=1}^{C}\exp(z^{weak}_i(k)/\tau)}, \ c=1,..,C
        \\
        y'_i = \arg\max_c p^{weak}_i(c), \ c=1,..,C
        \\
        p^{strong}_i(c) = \dfrac{\exp z^{strong}_i(c)}{\sum_{k=1}^{C}\exp(z^{strong}_i(k))},\ c=1,..,C
        \\
        L = -\dfrac{1}{b} \sum_{i=1}^{b}\mathbf{1}(p^{weak}(y'_i) \ge \tau) \cdot \log p^{strong}_i(y'_i)

    where :math:`z^{weak},z^{strong}` are the predictions of inputs with weak augmentation and strong augmentation respectively, and :math:`p^{weak},p^{strong}` are the probability distribution calculated from them. :math:`y'` is pseudo labels calculated from the predictions of inputs with weak augmentation.

    Args:
        threshold (float): The confidence threshold to accept pseudo_labels.
        t (float): The temperature used to sharpen the predictions.
        device(torch.torch.device): The device to put the result on.

    Inputs:
        - unlabeled_y_weak (tensor): The predictions of inputs with weak augmentation.
        - unlabeled_y_strong (tensor): The predictions of inputs with strong augmentation.

    Shape:
        - unlabeled_y_weak: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - unlabeled_y_strong: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - Output: scalar.
    """

    def __init__(self, threshold, t, device):
        super(FixMatchConsistencyLoss, self).__init__()
        self.threshold = threshold
        self.T = t
        self.device = device

    def forward(self, unlabeled_y_weak, unlabeled_y_strong):
        pseudo_label = torch.softmax(unlabeled_y_weak.detach() / self.T, dim=-1)
        max_prob, targets_unlabeled = torch.max(pseudo_label, dim=-1)
        mask = max_prob.ge(self.threshold).float()
        consistency_loss = (F.cross_entropy(unlabeled_y_strong, targets_unlabeled,
                                            reduction='none') * mask).mean()
        return consistency_loss
