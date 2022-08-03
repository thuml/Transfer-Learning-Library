"""
Modified from https://github.com/yxgeee/MMT
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_euclidean_distance(x, y):
    """Compute pairwise euclidean distance between two sets of features"""
    m, n = x.size(0), y.size(0)
    dist_mat = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t() \
               - 2 * torch.matmul(x, y.t())
    # for numerical stability
    dist_mat = dist_mat.clamp(min=1e-12).sqrt()
    return dist_mat


def hard_examples_mining(dist_mat, identity_mat, return_idxes=False):
    r"""Select hard positives and hard negatives according to `In defense of the Triplet Loss for Person
    Re-Identification (ICCV 2017) <https://arxiv.org/pdf/1703.07737v2.pdf>`_

    Args:
        dist_mat (tensor): pairwise distance matrix between two sets of features
        identity_mat (tensor): a matrix of shape :math:`(N, M)`. If two images :math:`P[i]` of set :math:`P` and
            :math:`Q[j]` of set :math:`Q` come from the same person, then :math:`identity\_mat[i, j] = 1`,
            otherwise :math:`identity\_mat[i, j] = 0`
        return_idxes (bool, optional): if True, also return indexes of hard examples. Default: False
    """
    # the implementation here is a little tricky, dist_mat contains pairwise distance between probe image and other
    # images in current mini-batch. As we want to select positive examples of the same person, we add a constant
    # negative offset on other images before sorting. As a result, images of the **same** person will rank first.
    sorted_dist_mat, sorted_idxes = torch.sort(dist_mat + (-1e7) * (1 - identity_mat), dim=1,
                                               descending=True)
    dist_ap = sorted_dist_mat[:, 0]
    hard_positive_idxes = sorted_idxes[:, 0]

    # the implementation here is similar to above code, we add a constant positive offset on images of same person
    # before sorting. Besides, we sort in ascending order. As a result, images of **different** persons will rank first.
    sorted_dist_mat, sorted_idxes = torch.sort(dist_mat + 1e7 * identity_mat, dim=1,
                                               descending=False)
    dist_an = sorted_dist_mat[:, 0]
    hard_negative_idxes = sorted_idxes[:, 0]
    if return_idxes:
        return dist_ap, dist_an, hard_positive_idxes, hard_negative_idxes
    return dist_ap, dist_an


class CrossEntropyLossWithLabelSmooth(nn.Module):
    r"""Cross entropy loss with label smooth from `Rethinking the Inception Architecture for Computer Vision
    (CVPR 2016) <https://arxiv.org/pdf/1512.00567.pdf>`_.

    Given one-hot labels :math:`labels \in R^C`, where :math:`C` is the number of classes,
    smoothed labels are calculated as

    .. math::
        smoothed\_labels = (1 - \epsilon) \times labels + \epsilon \times \frac{1}{C}

    We use smoothed labels when calculating cross entropy loss and this can be helpful for preventing over-fitting.

    Args:
        num_classes (int): number of classes.
        epsilon (float): a float number that controls the smoothness.

    Inputs:
        - y (tensor): unnormalized classifier predictions, :math:`y`
        - labels (tensor): ground truth labels, :math:`labels`

    Shape:
        - y: :math:`(minibatch, C)`, where :math:`C` is the number of classes
        - labels: :math:`(minibatch, )`
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLossWithLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, y, labels):
        log_prob = self.log_softmax(y)
        labels = torch.zeros_like(log_prob).scatter_(1, labels.unsqueeze(1), 1)
        labels = (1 - self.epsilon) * labels + self.epsilon / self.num_classes
        loss = (- labels * log_prob).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss augmented with batch hard from `In defense of the Triplet Loss for Person Re-Identification
    (ICCV 2017) <https://arxiv.org/pdf/1703.07737v2.pdf>`_.

    Args:
        margin (float): margin of triplet loss
        normalize_feature (bool, optional): if True, normalize features into unit norm first before computing loss.
            Default: False.
    """

    def __init__(self, margin, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

    def forward(self, f, labels):
        if self.normalize_feature:
            # equivalent to cosine similarity
            f = F.normalize(f)
        dist_mat = pairwise_euclidean_distance(f, f)

        n = dist_mat.size(0)
        identity_mat = labels.expand(n, n).eq(labels.expand(n, n).t()).float()

        dist_ap, dist_an = hard_examples_mining(dist_mat, identity_mat)
        y = torch.ones_like(dist_ap)
        loss = self.margin_loss(dist_an, dist_ap, y)
        return loss


class TripletLossXBM(nn.Module):
    r"""Triplet loss augmented with batch hard from `In defense of the Triplet Loss for Person Re-Identification
    (ICCV 2017) <https://arxiv.org/pdf/1703.07737v2.pdf>`_. The only difference from triplet loss lies in that
    both features from current mini batch and external storage (XBM) are involved.

    Args:
        margin (float, optional): margin of triplet loss. Default: 0.3
        normalize_feature (bool, optional): if True, normalize features into unit norm first before computing loss.
            Default: False

    Inputs:
        - f (tensor): features of current mini batch, :math:`f`
        - labels (tensor): identity labels for current mini batch, :math:`labels`
        - xbm_f (tensor): features collected from XBM, :math:`xbm\_f`
        - xbm_labels (tensor): corresponding identity labels of xbm_f, :math:`xbm\_labels`

    Shape:
        - f: :math:`(minibatch, F)`, where :math:`F` is the feature dimension
        - labels: :math:`(minibatch, )`
        - xbm_f: :math:`(minibatch, F)`
        - xbm_labels: :math:`(minibatch, )`
    """

    def __init__(self, margin=0.3, normalize_feature=False):
        super(TripletLossXBM, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, f, labels, xbm_f, xbm_labels):
        if self.normalize_feature:
            # equivalent to cosine similarity
            f = F.normalize(f)
            xbm_f = F.normalize(xbm_f)

        dist_mat = pairwise_euclidean_distance(f, xbm_f)

        # hard examples mining
        n, m = f.size(0), xbm_f.size(0)
        identity_mat = labels.expand(m, n).t().eq(xbm_labels.expand(n, m)).float()
        dist_ap, dist_an = hard_examples_mining(dist_mat, identity_mat)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


class SoftTripletLoss(nn.Module):
    r"""Soft triplet loss from `Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised
    Domain Adaptation on Person Re-identification (ICLR 2020) <https://arxiv.org/pdf/2001.01526.pdf>`_.
    Consider a triplet :math:`x,x_p,x_n` (anchor, positive, negative), corresponding features are :math:`f,f_p,f_n`.
    We optimize for a smaller distance between :math:`f` and :math:`f_p` and a larger distance
    between :math:`f` and :math:`f_n`. Inner product is adopted as their similarity measure, soft triplet loss is thus
    defined as

    .. math::
        loss = \mathcal{L}_{\text{bce}}(\frac{\text{exp}(f^Tf_p)}{\text{exp}(f^Tf_p)+\text{exp}(f^Tf_n)}, 1)

    where :math:`\mathcal{L}_{\text{bce}}` means binary cross entropy loss. We denote the first term in above loss function
    as :math:`T`. When features from another teacher network can be obtained, we can calculate :math:`T_{teacher}` as
    labels, resulting in the following soft version

    .. math::
        loss = \mathcal{L}_{\text{bce}}(T, T_{teacher})

    Args:
        margin (float, optional): margin of triplet loss. If None, soft labels from another network will be adopted when
            computing loss. Default: None.
        normalize_feature (bool, optional): if True, normalize features into unit norm first before computing loss.
            Default: False.
    """

    def __init__(self, margin=None, normalize_feature=False):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature

    def forward(self, features_1, features_2, labels):
        if self.normalize_feature:
            # equal to cosine similarity
            features_1 = F.normalize(features_1)
            features_2 = F.normalize(features_2)

        dist_mat = pairwise_euclidean_distance(features_1, features_1)
        assert dist_mat.size(0) == dist_mat.size(1)

        n = dist_mat.size(0)
        identity_mat = labels.expand(n, n).eq(labels.expand(n, n).t()).float()

        dist_ap, dist_an, ap_idxes, an_idxes = hard_examples_mining(dist_mat, identity_mat, return_idxes=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        if self.margin is not None:
            loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
            return loss

        dist_mat_ref = pairwise_euclidean_distance(features_2, features_2)
        dist_ap_ref = torch.gather(dist_mat_ref, 1, ap_idxes.view(n, 1).expand(n, n))[:, 0]
        dist_an_ref = torch.gather(dist_mat_ref, 1, an_idxes.view(n, 1).expand(n, n))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        loss = (- triple_dist_ref * triple_dist).sum(dim=1).mean()
        return loss


class CrossEntropyLoss(nn.Module):
    r"""We use :math:`C` to denote the number of classes, :math:`N` to denote mini-batch
    size, this criterion expects unnormalized predictions :math:`y\_{logits}` of shape :math:`(N, C)` and
    :math:`target\_{logits}` of the same shape :math:`(N, C)`. Then we first normalize them into
    probability distributions among classes

    .. math::
        y = \text{softmax}(y\_{logits})
    .. math::
        target = \text{softmax}(target\_{logits})

    Final objective is calculated as

    .. math::
        \text{loss} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^C -target_i^j \times \text{log} (y_i^j)
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, y, labels):
        log_prob = self.log_softmax(y)
        loss = (- F.softmax(labels, dim=1).detach() * log_prob).sum(dim=1).mean()
        return loss
