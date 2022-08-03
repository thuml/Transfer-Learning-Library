"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import os
from typing import Optional, Tuple, Any
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class MNIST(ImageList):
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
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/f18f1e115de94644b900/?dl=1"),
        ("mnist_train_image", "mnist_image.tar.gz", "https://cloud.tsinghua.edu.cn/f/fdf45c75d2e746acba93/?dl=1"),
        # ("mnist_test_image", "mnist_image.tar.gz", "https://cloud.tsinghua.edu.cn/f/fdf45c75d2e746acba93/?dl=1")
    ]
    image_list = {
        "train": "image_list/mnist_train.txt",
        "test": "image_list/mnist_test.txt"
    }
    CLASSES = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, mode="L", split='train', download: Optional[bool] = True, **kwargs):
        assert split in ['train', 'test']
        data_list_file = os.path.join(root, self.image_list[split])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        assert mode in ['L', 'RGB']
        self.mode = mode
        super(MNIST, self).__init__(root, MNIST.CLASSES, data_list_file=data_list_file, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index

        return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    @classmethod
    def get_classes(self):
        return MNIST.CLASSES


class USPS(ImageList):
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
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/10ddb319c24e40a08e58/?dl=1"),
        ("usps_train_image", "usps_image.tar.gz", "https://cloud.tsinghua.edu.cn/f/1d3d7e2540bd4392b715/?dl=1"),
        # ("usps_test_image", "usps_image.tar.gz", "https://cloud.tsinghua.edu.cn/f/1d3d7e2540bd4392b715/?dl=1")
    ]
    image_list = {
        "train": "image_list/usps_train.txt",
        "test": "image_list/usps_test.txt"
    }
    CLASSES = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, mode="L", split='train', download: Optional[bool] = True, **kwargs):
        assert split in ['train', 'test']
        data_list_file = os.path.join(root, self.image_list[split])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        assert mode in ['L', 'RGB']
        self.mode = mode
        super(USPS, self).__init__(root, USPS.CLASSES, data_list_file=data_list_file, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index

        return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target


class SVHN(ImageList):
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
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/80a8a06c4a324c59a5e4/?dl=1"),
        ("svhn_image", "svhn_image.tar.gz", "https://cloud.tsinghua.edu.cn/f/0e48a871e00345eb91a9/?dl=1")
    ]
    image_list = "image_list/svhn_balanced.txt"
    # image_list = "image_list/svhn.txt"
    CLASSES = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, mode="L", download: Optional[bool] = True, **kwargs):
        data_list_file = os.path.join(root, self.image_list)

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        assert mode in ['L', 'RGB']
        self.mode = mode
        super(SVHN, self).__init__(root, SVHN.CLASSES, data_list_file=data_list_file, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index

        return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target


class MNISTRGB(MNIST):
    def __init__(self, root, **kwargs):
        super(MNISTRGB, self).__init__(root, mode='RGB', **kwargs)


class USPSRGB(USPS):
    def __init__(self, root, **kwargs):
        super(USPSRGB, self).__init__(root, mode='RGB', **kwargs)


class SVHNRGB(SVHN):
    def __init__(self, root, **kwargs):
        super(SVHNRGB, self).__init__(root, mode='RGB', **kwargs)
