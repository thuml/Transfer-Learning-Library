from typing import Optional, Any
import os.path as osp
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
import random


class KITTIDist(ImageFolder):
    def __init__(self, root: str, split: str = 'train', download: Optional[bool] = True, **kwargs: Any):
        self.root = root
        if split == 'test':
            split = 'val'
        assert split in ("train", "val")
        if download and not osp.exists(osp.join(root, 'val')):
            download_and_extract_archive("https://cloud.tsinghua.edu.cn/f/b2386ad0a0b442569c58/?dl=1", root, root,
                                         filename='kitti_val.zip')
            download_and_extract_archive("https://cloud.tsinghua.edu.cn/f/cc7789849b7d49d1ad35/?dl=1", root,
                                         osp.join(root, "train"), filename='kitti_above_20.zip')
            download_and_extract_archive("https://cloud.tsinghua.edu.cn/f/31053d283e9e4aa9900f/?dl=1", root,
                                         osp.join(root, "train"), filename='kitti_below_8.zip')
            download_and_extract_archive("https://cloud.tsinghua.edu.cn/f/aa1adcbd463049368f5d/?dl=1", root,
                                         osp.join(root, "train"), filename='kitti_below_20.zip')
            download_and_extract_archive("https://cloud.tsinghua.edu.cn/f/eca1d767018b4eccbacb/?dl=1", root,
                                         osp.join(root, "train"), filename='no_vehicle.zip')

        super(KITTIDist, self).__init__(osp.join(root, split), **kwargs)
        random.seed(0)
        random.shuffle(self.samples)

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)
