"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from common.vision.datasets import ImageList


class Retinopathy(ImageList):
    """`Retinopathy <https://www.kaggle.com/c/diabetic-retinopathy-detection/data>`_ dataset \
        consists of image-label pairs with high-resolution retina images, and labels that indicate \
        the presence of Diabetic Retinopahy (DR) in a 0-4 scale (No DR, Mild, Moderate, Severe, \
        or Proliferative DR).

    .. note:: You need to download the source data manually into `root` directory.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    """
    CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    def __init__(self, root, split, download=False, **kwargs):

        super(Retinopathy, self).__init__(os.path.join(root, split), Retinopathy.CLASSES, os.path.join(root, "image_list", "{}.txt".format(split)), **kwargs)
