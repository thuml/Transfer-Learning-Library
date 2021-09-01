from typing import Optional, Any
from torchvision.datasets.utils import download_and_extract_archive
import os
import random

from torchvision.datasets.folder import ImageFolder


class DSpritesLocation(ImageFolder):
    def __init__(self, root: str, split: str = 'train', download: Optional[bool] = True, **kwargs: Any):
        self.root = root
        if split == 'test':
            split = 'val'
        assert split in ("train", "val")
        if download and not os.path.exists(os.path.join(root, 'dsprites_loc')):
            download_and_extract_archive("https://cloud.tsinghua.edu.cn/f/2ea29596f0a44dc898ab/?dl=1", root, root, filename='dsprites_loc.tgz')
        super(DSpritesLocation, self).__init__(os.path.join(root, 'dsprites_loc', split), **kwargs)
        random.seed(0)
        random.shuffle(self.samples)

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)


class DSpritesOrientation(ImageFolder):
    def __init__(self, root: str, split: str = 'train', download: Optional[bool] = True, **kwargs: Any):
        self.root = root
        if split == 'test':
            split = 'val'
        assert split in ("train", "val")
        if download and not os.path.exists(os.path.join(root, 'dsprites_orient')):
            download_and_extract_archive("https://cloud.tsinghua.edu.cn/f/b04db3f4bf9e401db650/?dl=1", root, root, filename='dsprites_orient.tgz')
        super(DSpritesOrientation, self).__init__(os.path.join(root, 'dsprites_orient', split), **kwargs)
        random.seed(0)
        random.shuffle(self.samples)

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)