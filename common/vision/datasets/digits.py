import torchvision.datasets as D
from PIL import Image
from typing import Tuple, Any
import numpy as np


class MNIST(D.MNIST):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (str): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        mode (str): The channel mode for image. Choices includes ``"L"```, ``"RGB"``.
            Default: ``"L"```
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, mode="L", split='train', **kwargs):
        assert mode in ['L', 'RGB']
        assert split in ['train', 'test']
        super(MNIST, self).__init__(root, train=split == 'train', **kwargs)
        self.mode = mode

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L').convert(self.mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class USPS(D.USPS):
    """`USPS <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps>`_ Dataset.
        The data-format is : [label [index:value ]*256 \\n] * num_lines, where ``label`` lies in ``[1, 10]``.
        The value for each pixel lies in ``[-1, 1]``. Here we transform the ``label`` into ``[0, 9]``
        and make pixel values in ``[0, 255]``.

        Args:
            root (str): Root directory of dataset to store``USPS`` data files.
            mode (str): The channel mode for image. Choices includes ``"L"```, ``"RGB"``.
                Default: ``"L"```
            split (str, optional): The dataset split, supports ``train``, or ``test``.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.

        """
    def __init__(self, root, mode="L", split='train', **kwargs):
        assert mode in ['L', 'RGB']
        assert split in ['train', 'test']
        super(USPS, self).__init__(root, train=split == 'train', **kwargs)
        self.mode = mode

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L').convert(self.mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHN(D.SVHN):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (str): Root directory of dataset where directory
            ``SVHN`` exists.
        mode (str): The channel mode for image. Choices includes ``"L"```, ``"RGB"``.
            Default: ``"RGB"```
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    def __init__(self, root, mode="RGB", **kwargs):
        super(SVHN, self).__init__(root, **kwargs)
        assert mode in ['L', 'RGB']
        self.mode = mode

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert(self.mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target