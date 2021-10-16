"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from common.vision.datasets import ImageList
from ._util import download as download_data, check_exits


class EuroSAT(ImageList):
    """
    `EuroSAT <https://github.com/phelber/eurosat>`_ dataset consists in classifying \
        Sentinel-2 satellite images into 10 different types of land use (Residential, \
        Industrial, River, Highway, etc). \
        The spatial resolution corresponds to 10 meters per pixel, and the image size \
        is 64x64 pixels.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    CLASSES =['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture',
                   'PermanentCrop', 'Residential', 'River', 'SeaLake']

    def __init__(self, root, split='train', download=False, **kwargs):
        if download:
            download_data(root, "eurosat", "eurosat.tgz", "https://cloud.tsinghua.edu.cn/f/9983d7ab86184d74bb17/?dl=1")
        else:
            check_exits(root, "eurosat")
        split = 'train[:21600]' if split == 'train' else 'train[21600:]'

        root = os.path.join(root, "eurosat")
        super(EuroSAT, self).__init__(root, EuroSAT.CLASSES, os.path.join(root, "imagelist", "{}.txt".format(split)), **kwargs)



