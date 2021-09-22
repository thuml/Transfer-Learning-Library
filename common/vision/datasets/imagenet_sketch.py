"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import os
from torchvision.datasets.imagenet import ImageNet
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class ImageNetSketch(ImageList):
    """ImageNet-Sketch Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: You need to put ``train`` directory, ``metabin`` of ImageNet-1K and ``sketch`` directory of ImageNet-Sketch
        manually in `root` directory.

        DALIB will only download ImageList automatically.
        In `root`, there will exist following files after preparing.
        ::
            metabin (from ImageNet)
            train/
                n02128385/
                ...
            val/
            sketch/
                n02128385/
            image_list/
                imagenet-train.txt
                sketch.txt
                ...
    """
    # TODO
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/8e12fff7b6224e5fa62b/?dl=1"),
    ]
    image_list = {
        "IN": "image_list/imagenet-train.txt",
        "IN-val": "image_list/imagenet-val.txt",
        "sketch": "image_list/sketch.txt",
    }

    def __init__(self, root: str, task: str, split: Optional[str] = 'all', download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        assert split in ["train", "val", "all"]
        if task == "IN" and split == "val":
            task = "IN-val"

        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(ImageNetSketch, self).__init__(root, ImageNet(root).classes, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())