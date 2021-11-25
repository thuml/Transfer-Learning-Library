"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
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
            Art/
                Alarm_Clock/*.jpg
                ...
            Clipart/
            Product/
            Real_World/
            image_list/
                Art.txt
                Clipart.txt
                Product.txt
                Real_World.txt
    """
    download_list = [       
    ]
    image_list = {
        "rareplanes": "image_list/rareplanes.txt",
        "fgvc": "image_list/fgvc.txt",
    }
    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

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