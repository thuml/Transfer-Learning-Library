"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, Sequence
import os
from .._util import download as download_data, check_exits
from .image_regression import ImageRegression


class MPI3D(ImageRegression):
    """`MPI3D <https://arxiv.org/abs/1906.03292>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'C'``: Color, \
            ``'N'``: Noisy and ``'S'``: Scream.
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        factors (sequence[str]): Factors selected. Default: ('horizontal axis', 'vertical axis').
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            real/
                ...
            realistic/
            toy/
            image_list/
                real_train.txt
                realistic_train.txt
                toy_train.txt
                real_test.txt
                realistic_test.txt
                toy_test.txt
        """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/f0ff24df967b42479d9e/?dl=1"),
        ("real", "real.tgz", "https://cloud.tsinghua.edu.cn/f/04c1318555fc4283862b/?dl=1"),
        ("realistic", "realistic.tgz", "https://cloud.tsinghua.edu.cn/f/2c0f7dacc73148cea593/?dl=1"),
        ("toy", "toy.tgz", "https://cloud.tsinghua.edu.cn/f/6327912a50374e20af95/?dl=1"),
    ]
    image_list = {
        "RL": "real",
        "RC": "realistic",
        "T": "toy"
    }
    FACTORS = ('horizontal axis', 'vertical axis')

    def __init__(self, root: str, task: str, split: Optional[str] = 'train',
                 factors: Sequence[str] = ('horizontal axis', 'vertical axis'),
                 download: Optional[bool] = True, target_transform=None, **kwargs):
        assert task in self.image_list
        assert split in ['train', 'test']
        for factor in factors:
            assert factor in self.FACTORS

        factor_index = [self.FACTORS.index(factor) for factor in factors]

        if target_transform is None:
            target_transform = lambda x: x[list(factor_index)] / 39.
        else:
            target_transform = lambda x: target_transform(x[list(factor_index)]) / 39.

        data_list_file = os.path.join(root, "image_list", "{}_{}.txt".format(self.image_list[task], split))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(MPI3D, self).__init__(root, factors, data_list_file=data_list_file, target_transform=target_transform, **kwargs)

