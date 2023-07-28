"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class VisDA2017(ImageList):
    """`VisDA-2017 <http://ai.bu.edu/visda-2017/assets/attachments/VisDA_2017.pdf>`_ Dataset

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'Synthetic'``: synthetic images and \
            ``'Real'``: real-world images.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            train/
                aeroplance/
                    *.png
                    ...
            validation/
            image_list/
                train.txt
                validation.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/c107de37b8094c5398dc/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/c5f3ce59139144ec8221/?dl=1"),
        ("validation", "validation.tgz", "https://cloud.tsinghua.edu.cn/f/da70e4b1cf514ecea562/?dl=1")
    ]
    image_list = {
        "Synthetic": "image_list/train.txt",
        "Real": "image_list/validation.txt"
    }
    CLASSES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
               'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(VisDA2017, self).__init__(root, VisDA2017.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())