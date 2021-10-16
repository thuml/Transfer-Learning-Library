from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class PACS(ImageList):
    """`PACS Dataset <https://domaingeneralization.github.io/#data>`_.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            art_painting/
                dog/
                    *.jpg
                    ...
            cartoon/
            photo/
            sketch
            image_list/
                art_painting.txt
                cartoon.txt
                photo.txt
                sketch.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/add42cc3859847bc988c/?dl=1"),
        ("art_painting", "art_painting.tgz", "https://cloud.tsinghua.edu.cn/f/4eb7db4f3eec41719856/?dl=1"),
        ("cartoon", "cartoon.tgz", "https://cloud.tsinghua.edu.cn/f/d847ac22497b4826889f/?dl=1"),
        ("photo", "photo.tgz", "https://cloud.tsinghua.edu.cn/f/458ad21483da4a45935b/?dl=1"),
        ("sketch", "sketch.tgz", "https://cloud.tsinghua.edu.cn/f/c892ac2d94a44b1196b8/?dl=1"),
    ]
    image_list = {
        "A": "image_list/art_painting_{}.txt",
        "C": "image_list/cartoon_{}.txt",
        "P": "image_list/photo_{}.txt",
        "S": "image_list/sketch_{}.txt"
    }
    CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    def __init__(self, root: str, task: str, split='all', download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        assert split in ["train", "val", "all", "test"]
        if split == "test":
            split = "all"
        data_list_file = os.path.join(root, self.image_list[task].format(split))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(PACS, self).__init__(root, PACS.CLASSES, data_list_file=data_list_file, target_transform=lambda x: x - 1,
                                   **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
