"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from torchvision.datasets.cifar import CIFAR10 as CIFAR10Base, CIFAR100 as CIFAR100Base


class CIFAR10(CIFAR10Base):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """

    def __init__(self, root, split='train', transform=None, download=True):
        super(CIFAR10, self).__init__(root, train=split == 'train', transform=transform, download=download)
        self.num_classes = 10


class CIFAR100(CIFAR100Base):
    """
    `CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """

    def __init__(self, root, split='train', transform=None, download=True):
        super(CIFAR100, self).__init__(root, train=split == 'train', transform=transform, download=download)
        self.num_classes = 100
