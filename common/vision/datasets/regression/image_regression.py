"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from typing import Optional, Callable, Tuple, Any, List, Sequence
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import numpy as np


class ImageRegression(datasets.VisionDataset):
    """A generic Dataset class for domain adaptation in image regression

    Args:
        root (str): Root directory of dataset
        factors (sequence[str]): Factors selected. Default: ('scale', 'position x', 'position y').
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note::
        In `data_list_file`, each line has `1+len(factors)` values in the following format.
        ::
            source_dir/dog_xxx.png x11, x12, ...
            source_dir/cat_123.png x21, x22, ...
            target_dir/dog_xxy.png x31, x32, ...
            target_dir/cat_nsdf3.png x41, x42, ...

        The first value is the relative path of an image, and the rest values are the ground truth of the corresponding factors.
        If your data_list_file has different formats, please over-ride :meth:`ImageRegression.parse_data_file`.
    """
    def __init__(self, root: str, factors: Sequence[str], data_list_file: str,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.factors = factors
        self.loader = default_loader
        self.data_list_file = data_list_file

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[float]]:
        """
        Args:
            index (int): Index

        Returns:
            (image, target) where target is a numpy float array.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, Any]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file

        Returns:
            List of (image path, (factors)) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                data = line.split()
                path = str(data[0])
                target = np.array([float(d) for d in data[1:]], dtype=np.float)
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                data_list.append((path, target))
        return data_list

    @property
    def num_factors(self) -> int:
        return len(self.factors)