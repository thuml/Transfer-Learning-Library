"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from collections import Counter

import torch


def convert_dataset(dataset):
    """
    Converts a dataset which returns (img, label) pairs into one that returns (index, img, label) triplets.
    """

    class DatasetWrapper:

        def __init__(self):
            self.dataset = dataset

        def __getitem__(self, index):
            return index, self.dataset[index]

        def __len__(self):
            return len(self.dataset)

    return DatasetWrapper()


class DynamicThresholdingModule(object):
    r"""
    Dynamic thresholding module from `FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling
    <https://arxiv.org/abs/2110.08263>`_.
    """

    def __init__(self, num_classes, n_unlabeled_samples, warmup, device):
        self.num_classes = num_classes
        self.n_unlabeled_samples = n_unlabeled_samples
        self.warmup = warmup
        self.net_outputs = torch.zeros(n_unlabeled_samples, dtype=torch.long).to(device)
        self.net_outputs.fill_(-1)
        self.device = device

    def get_status(self):
        pseudo_counter = Counter(self.net_outputs.tolist())
        if max(pseudo_counter.values()) == self.n_unlabeled_samples:
            # In the early stage of training, the network does not output pseudo labels with high confidence.
            # In this case, the learning status of all categories is simply zero.
            status = torch.zeros(self.n_unlabeled_samples).to(self.device)
            return status
        if not self.warmup and -1 in pseudo_counter.keys():
            pseudo_counter.pop(-1)
        max_num = max(pseudo_counter.values())
        status = [
            pseudo_counter[c] / max_num for c in range(self.num_classes)
        ]
        status = torch.FloatTensor(status).to(self.device)
        return status

    def update(self, idxes, selected_mask, pseudo_labels):
        if idxes[selected_mask == 1].nelement() != 0:
            self.net_outputs[idxes[selected_mask == 1]] = pseudo_labels[selected_mask == 1]
