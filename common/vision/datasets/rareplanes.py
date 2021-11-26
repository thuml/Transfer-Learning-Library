"""
@author: Hao Chen
@contact: chanhal@outlook.com
"""
import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Rareplanes(ImageList):
    """`OfficeHome <http://hemanthdv.org/OfficeHome-Dataset/>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
            ``'Cl'``: Clipart, ``'Pr'``: Product and ``'Rw'``: Real_World.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            rareplanes/
                Alarm_Clock/*.jpg
                ...
            fgvc/
            image_list/
                rareplanes.txt
                fgvc.txt
    """
    download_list = [       
    ]
    image_list = {
        "rareplanes": "image_list/rareplanes.txt",
        "fgvc": "image_list/fgvc.txt",
    }
    CLASSES = ['707-320','Boeing_717','727-200','737-200','737-300','747-200','757-300','767-200',
                '767-400','777-300','A300B4','A319','A320','A330-300','A340-300','A380','Cessna_172',
                'BAE_146-300','Fokker_100','MD-11']

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Rareplanes, self).__init__(root, Rareplanes.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())