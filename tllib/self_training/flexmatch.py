"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from collections import Counter

import torch


class DynamicThresholdingModule(object):
    r"""
    Dynamic thresholding module from `FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling
    <https://arxiv.org/abs/2110.08263>`_. At time :math:`t`, for each category :math:`c`,
    the learning status :math:`\sigma_t(c)` is estimated by the number of samples whose predictions fall into this class
    and above a threshold (e.g. 0.95). Then, FlexMatch normalizes :math:`\sigma_t(c)` to make its range between 0 and 1

    .. math::
        \beta_t(c) = \frac{\sigma_t(c)}{\underset{c'}{\text{max}}~\sigma_t(c')}.

    The dynamic threshold is formulated as

    .. math::
        \mathcal{T}_t(c) = \mathcal{M}(\beta_t(c)) \cdot \tau,

    where \tau denotes the pre-defined threshold (e.g. 0.95), :math:`\mathcal{M}` denotes a (possibly non-linear)
    mapping function.

    Args:
        threshold (float): The pre-defined confidence threshold
        warmup (bool): Whether perform threshold warm-up. If True, the number of unlabeled data that have not been
            used will be considered when normalizing :math:`\sigma_t(c)`
        mapping_func (callable): An increasing mapping function. For example, this function can be (1) concave
            :math:`\mathcal{M}(x)=\text{ln}(x+1)/\text{ln}2`, (2) linear :math:`\mathcal{M}(x)=x`,
            and (3) convex :math:`\mathcal{M}(x)=2/2-x`
        num_classes (int): Number of classes
        n_unlabeled_samples (int): Size of the unlabeled dataset
        device (torch.device): Device

    """

    def __init__(self, threshold, warmup, mapping_func, num_classes, n_unlabeled_samples, device):
        self.threshold = threshold
        self.warmup = warmup
        self.mapping_func = mapping_func
        self.num_classes = num_classes
        self.n_unlabeled_samples = n_unlabeled_samples
        self.net_outputs = torch.zeros(n_unlabeled_samples, dtype=torch.long).to(device)
        self.net_outputs.fill_(-1)
        self.device = device

    def get_threshold(self, pseudo_labels):
        """Calculate and return dynamic threshold"""
        pseudo_counter = Counter(self.net_outputs.tolist())
        if max(pseudo_counter.values()) == self.n_unlabeled_samples:
            # In the early stage of training, the network does not output pseudo labels with high confidence.
            # In this case, the learning status of all categories is simply zero.
            status = torch.zeros(self.num_classes).to(self.device)
        else:
            if not self.warmup and -1 in pseudo_counter.keys():
                pseudo_counter.pop(-1)
            max_num = max(pseudo_counter.values())
            # estimate learning status
            status = [
                pseudo_counter[c] / max_num for c in range(self.num_classes)
            ]
            status = torch.FloatTensor(status).to(self.device)
        # calculate dynamic threshold
        dynamic_threshold = self.threshold * self.mapping_func(status[pseudo_labels])
        return dynamic_threshold

    def update(self, idxes, selected_mask, pseudo_labels):
        """Update the learning status

        Args:
            idxes (tensor): Indexes of corresponding samples
            selected_mask (tensor): A binary mask, a value of 1 indicates the prediction for this sample will be updated
            pseudo_labels (tensor): Network predictions

        """
        if idxes[selected_mask == 1].nelement() != 0:
            self.net_outputs[idxes[selected_mask == 1]] = pseudo_labels[selected_mask == 1]
