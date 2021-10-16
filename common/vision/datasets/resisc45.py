"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""

from torchvision.datasets.folder import ImageFolder
import random


class Resisc45(ImageFolder):
    """`Resisc45 <http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html>`_ dataset \
        is a scene classification task from remote sensing images. There are 45 classes, \
        containing 700 images each, including tennis court, ship, island, lake, \
        parking lot, sparse residential, or stadium. \
        The image size is RGB 256x256 pixels.

    .. note:: You need to download the source data manually into `root` directory.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    """
    def __init__(self, root, split='train', download=False, **kwargs):
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
