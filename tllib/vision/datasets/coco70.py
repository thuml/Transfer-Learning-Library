"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class COCO70(ImageList):
    """COCO-70 dataset is a large-scale classification dataset (1000 images per class) created from
    `COCO <https://cocodataset.org/>`_ Dataset.
    It is used to explore the effect of fine-tuning with a large amount of data.

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
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d2ffb62fe3d140f1a73c/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/e0dc4368342948c5bb2a/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/59393a55c818429fb8d1/?dl=1"),
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
    CLASSES =['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
              'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'skis', 'kite', 'baseball_bat', 'skateboard', 'surfboard',
              'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
              'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'teddy_bear']

    def __init__(self, root: str, split: str, sample_rate: Optional[int] =100, download: Optional[bool] = False, **kwargs):

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

        super(COCO70, self).__init__(root, COCO70.CLASSES, data_list_file=data_list_file, **kwargs)
