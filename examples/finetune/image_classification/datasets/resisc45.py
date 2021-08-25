from torchvision.datasets.folder import ImageFolder
import random


class Resisc45(ImageFolder):
    def __init__(self, root, split='train', **kwargs):
        super(Resisc45, self).__init__(root, **kwargs)
        random.seed(0)
        random.shuffle(self.samples)
        if split == 'train':
            self.samples = self.samples[:25200]
        else:
            self.samples = self.samples[25200:]

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)
