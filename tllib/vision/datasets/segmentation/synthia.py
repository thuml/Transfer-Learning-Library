"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from .segmentation_list import SegmentationList
from .cityscapes import Cityscapes
from .._util import download as download_data


class Synthia(SegmentationList):
    """`SYNTHIA <https://synthia-dataset.net/>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``.
        data_folder (str, optional): Sub-directory of the image. Default: 'RGB'.
        label_folder (str, optional): Sub-directory of the label. Default: 'synthia_mapped_to_cityscapes'.
        mean (seq[float]): mean BGR value. Normalize the image if not None. Default: None.
        transforms (callable, optional): A function/transform that  takes in  (PIL image, label) pair \
            and returns a transformed version. E.g, :class:`~tllib.vision.transforms.segmentation.Resize`.

    .. note:: You need to download GTA5 manually.
        Ensure that there exist following directories in the `root` directory before you using this class.
        ::
            RGB/
            synthia_mapped_to_cityscapes/
    """
    ID_TO_TRAIN_ID = {
        3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
        15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
        8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18
    }
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/15c4d0f8e62e45d9a6b7/?dl=1"),
    ]

    def __init__(self, root, split='train', data_folder='RGB', label_folder='synthia_mapped_to_cityscapes', **kwargs):
        assert split in ['train']
        # download meta information from Internet
        list(map(lambda args: download_data(root, *args), self.download_list))
        data_list_file = os.path.join(root, "image_list", "{}.txt".format(split))
        super(Synthia, self).__init__(root, Cityscapes.CLASSES, data_list_file, data_list_file, data_folder,
                                      label_folder, id_to_train_id=Synthia.ID_TO_TRAIN_ID,
                                      train_id_to_color=Cityscapes.TRAIN_ID_TO_COLOR, **kwargs)

    @property
    def evaluate_classes(self):
        return [
            'road', 'sidewalk', 'building', 'traffic light', 'traffic sign',
            'vegetation', 'sky', 'person', 'rider', 'car', 'bus', 'motorcycle', 'bicycle'
        ]
