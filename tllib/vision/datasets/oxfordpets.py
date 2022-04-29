"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OxfordIIITPets(ImageList):
    """`The Oxford-IIIT Pets <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ \
    is a 37-category pet dataset with roughly 200 images for each class.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        sample_rate (int): The sampling rates to sample random ``training`` images for each category.
            Choices include 100, 50, 30, 15. Default: 100.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            train/
            test/
            image_list/
                train_100.txt
                train_50.txt
                train_30.txt
                train_15.txt
                test.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/8b7fb79279174bcc8558/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/e333a09b93a34a0ebef6/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/ce00352d79c34ea48bf4/?dl=1"),
    ]
    image_list = {
        "train": "image_list/train_100.txt",
        "train100": "image_list/train_100.txt",
        "train50": "image_list/train_50.txt",
        "train30": "image_list/train_30.txt",
        "train15": "image_list/train_15.txt",
        "test": "image_list/test.txt",
        "test100": "image_list/test.txt",
    }
    CLASSES = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal',
               'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel',
               'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger',
               'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll',
               'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx',
               'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

    def __init__(self, root: str, split: str, sample_rate: Optional[int] = 100, download: Optional[bool] = False,
                 **kwargs):

        if split == 'train':
            list_name = 'train' + str(sample_rate)
            assert list_name in self.image_list
            data_list_file = os.path.join(root, self.image_list[list_name])
        else:
            data_list_file = os.path.join(root, self.image_list['test'])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(OxfordIIITPets, self).__init__(root, OxfordIIITPets.CLASSES, data_list_file=data_list_file, **kwargs)
