"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from .segmentation_list import SegmentationList
from .cityscapes import Cityscapes
from .._util import download as download_data


class GTA5(SegmentationList):
    """`GTA5 <https://download.visinf.tu-darmstadt.de/data/from_games/>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``.
        data_folder (str, optional): Sub-directory of the image. Default: 'images'.
        label_folder (str, optional): Sub-directory of the label. Default: 'labels'.
        mean (seq[float]): mean BGR value. Normalize the image if not None. Default: None.
        transforms (callable, optional): A function/transform that  takes in  (PIL image, label) pair \
            and returns a transformed version. E.g, :class:`~common.vision.transforms.segmentation.Resize`.

    .. note:: You need to download GTA5 manually.
        Ensure that there exist following directories in the `root` directory before you using this class.
        ::
            images/
            labels/
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/c77ff6fc4eea435791f4/?dl=1"),
    ]

    def __init__(self, root, split='train', data_folder='images', label_folder='labels', **kwargs):
        assert split in ['train']
        # download meta information from Internet
        list(map(lambda args: download_data(root, *args), self.download_list))
        data_list_file = os.path.join(root, "image_list", "{}.txt".format(split))
        self.split = split
        super(GTA5, self).__init__(root, Cityscapes.CLASSES, data_list_file, data_list_file, data_folder, label_folder,
                                   id_to_train_id=Cityscapes.ID_TO_TRAIN_ID, train_id_to_color=Cityscapes.TRAIN_ID_TO_COLOR, **kwargs)