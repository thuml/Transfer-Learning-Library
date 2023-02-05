"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class SUN397(ImageList):
    """`SUN397 <https://vision.princeton.edu/projects/2010/SUN/>`_  is a dataset for scene understanding
    with 108,754 images in 397 scene categories. The number of images varies across categories,
    but there are at least 100 images per category. Note that the authors construct 10 partitions,
    where each partition contains 50 training images and 50 testing images per class. We adopt partition 1.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    """
    dataset_url = ("SUN397", "SUN397.tar.gz", "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz")
    image_list_url = (
        "SUN397/image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/17399c6e0455480aa174/?dl=1")

    def __init__(self, root, split='train', download=True, **kwargs):
        if download:
            download_data(root, *self.dataset_url)
            download_data(os.path.join(root, 'SUN397'), *self.image_list_url)
        else:
            check_exits(root, "SUN397")
            check_exits(root, "SUN397/image_list")

        classes = list([str(i) for i in range(397)])
        root = os.path.join(root, 'SUN397')
        super(SUN397, self).__init__(root, classes, os.path.join(root, 'image_list', '{}.txt'.format(split)), **kwargs)
