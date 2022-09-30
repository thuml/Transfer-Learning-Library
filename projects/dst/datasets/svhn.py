"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import numpy as np
from torchvision.datasets.svhn import SVHN as SVHNBase


class SVHN(SVHNBase):
    """
    `SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    """

    def __init__(self, root, split='train', transform=None, download=True):
        super(SVHN, self).__init__(root, split=split, transform=transform, download=download)
        self.num_classes = 10
        self.targets = self.labels.astype(np.int32)
