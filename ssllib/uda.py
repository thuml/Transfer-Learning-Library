"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output


class SupervisedUDALoss(nn.Module):
    r"""
    The supervised loss from `Unsupervised Data Augmentation for Consistency Training (NIPS 2020)
    <https://proceedings.neurips.cc/paper/2020/file/44feb0096faa8326192570788b38c1d1-Paper.pdf>`_.
    It can be described as

    .. math::
        p_i(c) = \dfrac{\exp (z_i(c))}{\sum_{k=1}^{C}\exp(z_i(k))}, \ c=1,..,C
        \\ \mathrm{mask}_i = \mathbf{1}[\max_c p_i(c) < \mathrm{threshold}]
        \\ L = - \dfrac{1}{\max(1,\sum_{i=1}^{b} \mathrm{mask}_i)}\sum_{i=1}^{b} \mathrm{mask}_i \cdot \log p_i(y_i)

    where :math:`z` is the predictions of labeled samples, :math:`p` is the probability distribution and :math:`y` is the ground truth labels.

    Args:
        model (torch.nn.Module): The model to calculate the loss.
        device(torch.torch.device): The device to put the result on.

    Inputs:
        - labeled_y (tensor): The predictions of labeled inputs.
        - label (tensor): The ground truth labels.
        - tsa_thresh (float): the threshold to discard the samples.

    Shape:
        - labeled_y: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - label: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - Output: scalar.
    """

    def __init__(self, model, device):
        super(SupervisedUDALoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.model = model
        self.device = device

    def forward(self, labeled_y, label, tsa_thresh):
        supervised_probability = F.softmax(labeled_y, dim=1)
        one_hot_label = F.one_hot(label, num_classes=labeled_y.shape[1])
        correct_label_probability = torch.sum(one_hot_label * supervised_probability, dim=-1)
        larger_than_threshold = correct_label_probability > tsa_thresh
        loss_mask = torch.ones_like(label, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
        loss_mask = loss_mask.detach()
        supervised_loss = self.criterion(labeled_y, label) * loss_mask
        supervised_loss = torch.sum(supervised_loss, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1),
                                                                         torch.tensor(1.).to(self.device))
        return supervised_loss


class UnsupervisedUDALoss(nn.Module):
    r"""
    The unsupervised loss from `Unsupervised Data Augmentation for Consistency Training (NIPS 2020) <https://proceedings.neurips.cc/paper/2020/file/44feb0096faa8326192570788b38c1d1-Paper.pdf>`_.
    It can be described as

    .. math::
        p^{weak}_i(c) = \dfrac{\exp (z^{weak}_i(c)/\tau)}{\sum_{k=1}^{C}\exp(z^{weak}_i(k)/\tau)}, \ c=1,..,C
        \\z^{ori}_i = f(x^{ori})_i, \ c=1,..,C
        \\p^{ori}_i(c) = \dfrac{\exp (z^{ori}_i(c))}{\sum_{k=1}^{C}\exp(z^{ori}_i(k))}, \ c=1,..,C
        \\ \mathrm{mask}_i = \mathbf{1}[\max_c p^{ori}_i(c) > \mathrm{threshold}]
        \\p^{aug}_i(c) = \dfrac{\exp (z^{aug}_i(c)/\tau)}{\sum_{k=1}^{C}\exp(z^{aug}_i(k)/\tau)}, \ c=1,..,C
        \\ L = \dfrac{1}{\max(1,\sum_{i=1}^{b} \mathrm{mask}_i)}
        \sum_{i=1}^{b}\mathrm{mask}_i \cdot \sum_{c=1}^{C} \ p^{ori}_i(c)\log \dfrac{p^{ori}_i(c)}{p^{aug}_i(c)}

    where :math:`f` is the model to calculate the loss, :math:`x^{ori}` is the original input without data augmentation, :math:`z^{aug}` is the predictions of inputs with data augmentation, and :math:`p^{ori},p^{aug}` are the probability distribution.

    Args:
        model (torch.nn.Module): The model to calculate the loss.
        uda_confidence_thresh (float): The confidence threshold to accept samples.
        t (float): The temperature used to sharpen the predictions.
        device(torch.torch.device): The device to put the result on.

    Inputs:
        - unlabeled_x_original (tensor): The input without augmentation fed to the model.
        - unlabeled_y_augmentation (tensor): The predictions of inputs with data augmentation.

    Shape:
        - unlabeled_x_original: :math:`(b, *)` where :math:`b` is the batch size, :math:`C` is the number of classes and * means any number of additional dimensions.
        - unlabeled_y_augmentation: :math:`(b, C)` where :math:`b` is the batch size and :math:`C` is the number of classes.
        - Output: scalar.
    """

    def __init__(self, model, uda_confidence_thresh, t, device):
        super(UnsupervisedUDALoss, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='none')
        self.model = model
        self.uda_confidence_thresh = uda_confidence_thresh
        self.T = t
        self.device = device

    def forward(self, unlabeled_x_original, unlabeled_y_augmentation):
        with torch.no_grad():
            unlabeled_original_y, f = self.model(unlabeled_x_original)
            original_probability = F.softmax(unlabeled_original_y, dim=-1)
            unsupervised_loss_mask = torch.max(original_probability, dim=-1)[0] > self.uda_confidence_thresh
            unsupervised_loss_mask = unsupervised_loss_mask.type(torch.float32)
            unsupervised_loss_mask = unsupervised_loss_mask.to(self.device)

        uda_softmax_temperature = self.T if self.T > 0 else 1.

        augmentation_log_probability = F.log_softmax(unlabeled_y_augmentation / uda_softmax_temperature, dim=-1)

        consistency_loss = torch.sum(self.criterion(augmentation_log_probability, original_probability), dim=-1)
        consistency_loss = torch.sum(consistency_loss * unsupervised_loss_mask, dim=-1) / torch.max(
            torch.sum(unsupervised_loss_mask, dim=-1), torch.tensor(1.).to(self.device))
        return consistency_loss
