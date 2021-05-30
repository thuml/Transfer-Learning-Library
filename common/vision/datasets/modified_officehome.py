import os
import torch
from typing import Optional, List
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class ModifiedOfficeHome(ImageList):
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

    .. note:: The objects are labeled i = C*S_i + C_i where C is the number of categories, C_i is the category label of the object, and S_i is the style index of the object. This is so you can retrieve the category of the the object (S_i = i % C) and the style of the object (C_i = i // C).

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
        ("image_list", "image_list.zip",
         "https://cloud.tsinghua.edu.cn/f/ca3a3b6a8d554905b4cd/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1"),
        ("Clipart", "Clipart.tgz",
         "https://cloud.tsinghua.edu.cn/f/0d41e7da4558408ea5aa/?dl=1"),
        ("Product", "Product.tgz",
         "https://cloud.tsinghua.edu.cn/f/76186deacd7c4fa0a679/?dl=1"),
        ("Real_World", "Real_World.tgz",
         "https://cloud.tsinghua.edu.cn/f/dee961894cc64b1da1d7/?dl=1")
    ]
    image_style_list = {
        "Ar": "image_list/Art.txt",
        "Cl": "image_list/Clipart.txt",
        "Pr": "image_list/Product.txt",
        "Rw": "image_list/Real_World.txt",
    }
    mod_image_style_list = {
        "Ar": "image_list/Modifed_Art.txt",
        "Cl": "image_list/Modified_Clipart.txt",
        "Pr": "image_list/Modified_Product.txt",
        "Rw": "image_list/Modified_Real_World.txt",
    }
    CATEGORIES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
                  'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
                  'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
                  'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
                  'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
                  'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
                  'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
    category_index = 0

    def __init__(self, root: str, tasks: List[str], download: Optional[bool] = False, **kwargs):
        # TODO: Incorporate modified style file list
        # TODO: Make it accept lists of styles
        # assert task in self.image_list
        mod_data_list_files = []
        data_list_files = []
        for task in tasks:
            assert task in self.image_style_list
            mod_data_list_files.append(
                os.path.join(root, self.mod_image_style_list[task])
            )
            data_list_files.append(
                os.path.join(root, self.image_style_list[task])
            )
        self.num_categories = len(ModifiedOfficeHome.CATEGORIES)
        self.num_styles = len(ModifiedOfficeHome.image_style_list)
        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(
                root, file_name), self.download_list))

        # check if the dataset file has been modified
        # modify the file with the new category-style label
        for i in range(len(mod_data_list_files)):
            file_name = data_list_files[i]
            new_contents = ""
            with open(file_name, "r") as f:
                for line in f.readlines():
                    split_line = line.split()
                    target = split_line[-1]
                    path = ' '.join(split_line[:-1])
                    target = int(target)
                    new_target = str(
                        target + ModifiedOfficeHome.category_index * self.num_categories
                    )
                    new_contents += path + ' ' + new_target + '\n'
            mod_file_name = mod_data_list_files[i]
            with open(mod_file_name, "w") as f:
                f.write(new_contents)
            ModifiedOfficeHome.category_index += 1

        super(ModifiedOfficeHome, self).__init__(
            # TODO: Adapt the code for predicting style instead of category
            root, ModifiedOfficeHome.CATEGORIES, data_list_files=mod_data_list_files, **kwargs
        )

    @classmethod
    def get_category(cls, labels: torch.tensor, num_categories: int) -> torch.tensor:
        return labels % num_categories

    @classmethod
    def get_style(cls, labels: torch.tensor, num_categories: int) -> torch.tensor:
        return labels // num_categories

    @classmethod
    def domains(cls):
        return list(cls.mod_image_style_list.keys())
