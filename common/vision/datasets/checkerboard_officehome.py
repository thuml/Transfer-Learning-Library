import os
import torch
import random
from typing import Optional, List
from .imagelist import ImageList, num_classes
from ._util import download as download_data, check_exits


class CheckerBoardOfficeHome(ImageList):
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
                train.txt
                test.txt
    """
    download_list = [
        # ("image_list", "image_list.zip",
        #  "https://cloud.tsinghua.edu.cn/f/ca3a3b6a8d554905b4cd/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1"),
        ("Clipart", "Clipart.tgz",
         "https://cloud.tsinghua.edu.cn/f/0d41e7da4558408ea5aa/?dl=1"),
        ("Product", "Product.tgz",
         "https://cloud.tsinghua.edu.cn/f/76186deacd7c4fa0a679/?dl=1"),
        ("Real_World", "Real_World.tgz",
         "https://cloud.tsinghua.edu.cn/f/dee961894cc64b1da1d7/?dl=1")
    ]
    images_dirs = {
        "Ar": "Art/",
        "Cl": "Clipart/",
        "Pr": "Product/",
        "Rw": "Real_World/",
    }
    # image_style_list = {
    #     "Ar": "image_list/Art.txt",
    #     "Cl": "image_list/Clipart.txt",
    #     "Pr": "image_list/Product.txt",
    #     "Rw": "image_list/Real_World.txt",
    # }
    # mod_image_style_list = {
    #     "Ar": "image_list/Modifed_Art.txt",
    #     "Cl": "image_list/Modified_Clipart.txt",
    #     "Pr": "image_list/Modified_Product.txt",
    #     "Rw": "image_list/Modified_Real_World.txt",
    # }

    # generate the train and test data together since they are disjoint sets that make up the set of all images
    image_lists = {
        "train": "image_list/train.txt",
        "novel": "image_list/novel.txt"
    }

    CATEGORIES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
                  'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
                  'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
                  'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
                  'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
                  'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
                  'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
    category_index = 0

    def __init__(self, root: str, tasks: List[str], download: Optional[bool] = False, style_is_domain: Optional[bool] = True, **kwargs):
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
        self.num_categories = len(CheckerBoardOfficeHome.CATEGORIES)
        self.num_styles = len(CheckerBoardOfficeHome.image_style_list)
        self.cat_style_matrix = torch.zeros((self.num_styles, self.num_categories))
        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(
                root, file_name), self.download_list))

        super(CheckerBoardOfficeHome, self).__init__(
            # TODO: Adapt the code for predicting style instead of category
            root, CheckerBoardOfficeHome.CATEGORIES, data_list_files=mod_data_list_files, **kwargs
        )

    def generate_image_list(self, num_train_styles: int):
        # TODO: Figure out 
        # TODO: Produce image list if style-predicting instead of category-predicting
        train_list = []
        novel_list = ""
        styles = CheckerBoardOfficeHome.image_style_list.keys()
        style_indices = list(range(self.num_styles))
        

        for cat_index in range(self.num_categories):
            random.shuffle(style_indices)
            for style_index in style_indices:
                style_count = 0
                image_dir = os.path.join(self.root,
                                         CheckerBoardOfficeHome.image_dirs[styles[style_index]],
                                         CheckerBoardOfficeHome.CATEGORIES[cat_index])
                for filename in os.listdir(image_dir):
                    if filename.endswith(".jpg"):
                        label = self._get_label(style_index, cat_index)
                        line = filename + ' ' + label + '\n'
                        if style_count < num_train_styles:
                            train_list += line
                            self.cat_style_matrix[style_index, cat_index] = 1
                        else:
                            novel_list += line
                style_count += 1

        train_list_filename = os.path.join(
            self.root, CheckerBoardOfficeHome.image_lists['train'])
        novel_list_filename = os.path.join(
            self.root, CheckerBoardOfficeHome.image_lists['novel'])
        with open(train_list_filename, "w") as f:
            f.write(train_list)
        with open(novel_list_filename, "w") as f:
            f.write(novel_list)

    def __str__(self):
        str_matrix = "Categories (Cols) AND Styles (Rows) Matrix\n "
        style_index = 0
        for cat_index in range(self.num_categories):
            str_matrix += "|" + cat_index
        str_matrix += "|\n"
        for row in self.cat_style_matrix:
            str_matrix += "|" + style_index
            for val in row:
                if val == 1:
                    str_matrix += "|X"
                else:
                    str_matrix += "| " 
            str_matrix += "|\n"
            style_index += 1
        return str_matrix

                

    def _get_label(self, style_index: int, category_index: int) -> int:
        return self.num_categories * style_index + category_index

    def domains(self):
        if self.style_is_domain:
            return list(self.images_dirs.keys())
        else:
            return CheckerBoardOfficeHome.CATEGORIES

    @classmethod
    def get_category(cls, labels: torch.tensor, num_categories: int) -> torch.tensor:
        return labels % num_categories

    @classmethod
    def get_style(cls, labels: torch.tensor, num_categories: int) -> torch.tensor:
        return labels // num_categories

    
