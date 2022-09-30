"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import numpy as np
from torchvision.datasets.stl10 import STL10 as STL10Base


class STL10(STL10Base):
    """
    `STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.
    """

    def __init__(self, root, split='train', transform=None, download=True):
        super(STL10, self).__init__(root, split=split, transform=transform, download=download)
        self.num_classes = 10
        self.targets = self.labels.astype(np.int32)
