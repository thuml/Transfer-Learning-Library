from torchvision.datasets.cifar import CIFAR10 as CIFAR10Base, CIFAR100 as CIFAR100Base


class CIFAR10(CIFAR10Base):
    def __init__(self, root, split='train', transform=None, download=True):
        super(CIFAR10, self).__init__(root, train=split=='train', transform=transform, download=download)
        self.num_classes = 10


class CIFAR100(CIFAR100Base):
    def __init__(self, root, split='train', transform=None, download=True):
        super(CIFAR100Base, self).__init__(root, train=split=='train', transform=transform, download=download)
        self.num_classes = 100