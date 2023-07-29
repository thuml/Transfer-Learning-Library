"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class DTD(ImageList):
    """
    `The Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html>`_ is an \
        evolving collection of textural images in the wild, annotated with a series of human-centric attributes, \
         inspired by the perceptual properties of textures. \
         The task consists in classifying images of textural patterns (47 classes, with 120 training images each). \
         Some of the textures are banded, bubbly, meshed, lined, or porous. \
         The image size ranges between 300x300 and 640x640 pixels.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/2218bfa61bac46539dd7/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/08fd47d35fc94f36a508/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/15873fe162c343cca8ed/?dl=1"),
        ("validation", "validation.tgz", "https://cloud.tsinghua.edu.cn/f/75c9ab22ebea4c3b87e7/?dl=1"),
    ]
    CLASSES = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked',
               'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy',
               'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled',
               'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous',
               'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped',
               'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']

    def __init__(self, root, split, download=False, **kwargs):
        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(DTD, self).__init__(root, DTD.CLASSES, os.path.join(root, "image_list", "{}.txt".format(split)), **kwargs)
