"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from common.vision.datasets import ImageList
from ._util import download as download_data, check_exits


class PatchCamelyon(ImageList):
    """
    The `PatchCamelyon <https://patchcamelyon.grand-challenge.org/>`_ dataset contains \
        327680 images of histopathologic scans of lymph node sections. \
        The classification task consists in predicting the presence of metastatic tissue \
         in given image (i.e., two classes). All images are 96x96 pixels

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    CLASSES = ['0', '1']

    def __init__(self, root, split, download=False,  **kwargs):
        if download:
            download_data(root, "patch_camelyon", "patch_camelyon.tgz", "https://cloud.tsinghua.edu.cn/f/21360b3441a54274b843/?dl=1")
        else:
            check_exits(root, "patch_camelyon")

        root = os.path.join(root, "patch_camelyon")
        super(PatchCamelyon, self).__init__(root, PatchCamelyon.CLASSES, os.path.join(root, "imagelist", "{}.txt".format(split)), **kwargs)

