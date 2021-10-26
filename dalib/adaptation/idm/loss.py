"""
Modified from https://github.com/SikaStar/IDM
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn


class DivLoss(nn.Module):
    """Diversity loss, which is defined as negative of standard deviation.
    """
    def __init__(self, ):
        super(DivLoss, self).__init__()

    def forward(self, lam):
        mu = lam.mean(0)
        std = ((lam - mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
        loss_std = -std.sum()
        return loss_std


class BridgeFeatLoss(nn.Module):
    """Bridge loss on feature space.
    """
    def __init__(self):
        super(BridgeFeatLoss, self).__init__()

    def forward(self, f_s, f_t, f_mixed, lam):
        dist_mixed2s = ((f_mixed - f_s) ** 2).sum(1, keepdim=True)
        dist_mixed2t = ((f_mixed - f_t) ** 2).sum(1, keepdim=True)

        dist_mixed2s = dist_mixed2s.clamp(min=1e-12).sqrt()
        dist_mixed2t = dist_mixed2t.clamp(min=1e-12).sqrt()

        dist_mixed = torch.cat((dist_mixed2s, dist_mixed2t), 1)
        lam_dist_mixed = (lam * dist_mixed).sum(1, keepdim=True)
        loss = lam_dist_mixed.mean()

        return loss


class BridgeProbLoss(nn.Module):
    """Bridge loss on prediction space.
    """
    def __init__(self, num_classes, epsilon=0.1):
        super(BridgeProbLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, y, labels, lam, device_num=4):
        y = y.view(device_num, -1, y.size(-1))
        y_s, y_t, y_mixed = y.split(y.size(1) // 3, dim=1)
        y_ori = torch.cat((y_s, y_t), 1).view(-1, y.size(-1))
        y_mixed = y_mixed.contiguous().view(-1, y.size(-1))
        log_prob_ori = self.log_softmax(y_ori)
        log_prob_mixed = self.log_softmax(y_mixed)

        labels = torch.zeros_like(log_prob_ori).scatter_(1, labels.unsqueeze(1), 1)
        labels = labels.view(device_num, -1, labels.size(-1))
        labels_s, labels_t = labels.split(labels.size(1) // 2, dim=1)
        labels_s = labels_s.contiguous().view(-1, labels.size(-1))
        labels_t = labels_t.contiguous().view(-1, labels.size(-1))

        labels = labels.view(-1, labels.size(-1))
        soft_labels = (1 - self.epsilon) * labels + self.epsilon / self.num_classes

        lam = lam.view(-1, 1)
        soft_labels_mixed = lam * labels_s + (1. - lam) * labels_t
        soft_labels_mixed = (1 - self.epsilon) * soft_labels_mixed + self.epsilon / self.num_classes
        loss_ori = (- soft_labels * log_prob_ori).sum(dim=1).mean()
        loss_bridge_prob = (- soft_labels_mixed * log_prob_mixed).sum(dim=1).mean()

        return loss_ori, loss_bridge_prob
