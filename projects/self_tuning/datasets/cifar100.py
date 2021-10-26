"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import numpy as np
from PIL import Image
from torchvision import datasets


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, idxes, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if idxes is not None:
            self.data = self.data[idxes]
            self.targets = np.array(self.targets)[idxes]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
