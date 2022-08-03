"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, Sequence
import os
from .._util import download as download_data, check_exits
from .image_regression import ImageRegression


class DSprites(ImageRegression):
    """`DSprites <https://github.com/deepmind/dsprites-dataset>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'C'``: Color, \
            ``'N'``: Noisy and ``'S'``: Scream.
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        factors (sequence[str]): Factors selected. Default: ('scale', 'position x', 'position y').
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            color/
                ...
            noisy/
            scream/
            image_list/
                color_train.txt
                noisy_train.txt
                scream_train.txt
                color_test.txt
                noisy_test.txt
                scream_test.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/fbbb6b1a43034712b34d/?dl=1"),
        ("color", "color.tgz", "https://cloud.tsinghua.edu.cn/f/9ce9f2abc61f49ed995a/?dl=1"),
        ("noisy", "noisy.tgz", "https://cloud.tsinghua.edu.cn/f/674435c8cb914ca0ad10/?dl=1"),
        ("scream", "scream.tgz", "https://cloud.tsinghua.edu.cn/f/0613675916ac4c3bb6bd/?dl=1"),
    ]
    image_list = {
        "C": "color",
        "N": "noisy",
        "S": "scream"
    }
    FACTORS = ('none', 'shape', 'scale', 'orientation', 'position x', 'position y')

    def __init__(self, root: str, task: str, split: Optional[str] = 'train',
                 factors: Sequence[str] = ('scale', 'position x', 'position y'),
                 download: Optional[bool] = True, target_transform=None, **kwargs):
        assert task in self.image_list
        assert split in ['train', 'test']
        for factor in factors:
            assert factor in self.FACTORS

        factor_index = [self.FACTORS.index(factor) for factor in factors]

        if target_transform is None:
            target_transform = lambda x: x[list(factor_index)]
        else:
            target_transform = lambda x: target_transform(x[list(factor_index)])

        data_list_file = os.path.join(root, "image_list", "{}_{}.txt".format(self.image_list[task], split))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(DSprites, self).__init__(root, factors, data_list_file=data_list_file, target_transform=target_transform, **kwargs)

