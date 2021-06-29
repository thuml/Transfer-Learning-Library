import os
import torch
import torchvision.transforms as T
import random
from itertools import combinations
from math import ceil
from typing import Optional, List, Tuple
from .imagelist import ImageList
from ._util import download as download_data, check_exits



class CheckerboardOfficeHome211():
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
                val.txt
                novel.txt
    """
    download_list = [
        ("Art", "Art.tgz",
         "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1"),
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

    images_lists = ("train.txt", "val.txt", "novel.txt")

    CATEGORIES = [
        'Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet',
        'Shelf', 'Toys', 'Sink', 'Laptop', 'Kettle', 'Folder', 'Keyboard',
        'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch', 'Bike',
        'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet',
        'Mouse', 'Pen', 'Monitor', 'Mop', 'Sneakers', 'Notebook', 'Backpack',
        'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio', 'Fan',
        'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker',
        'Eraser', 'Bucket', 'Chair', 'Calendar', 'Calculator', 'Flowers',
        'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
        'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator',
        'Marker'
    ]

    # has_gen_file_list = False
    num_categories = len(CATEGORIES)
    num_styles = len(images_dirs.keys())

    def __init__(self,
                root: str,
                download: Optional[bool] = False,
                balance_domains: Optional[bool] = False, 
                transforms = [None, None, None],
                 **kwargs):
        assert len(transforms) == len(CheckerboardOfficeHome211.images_lists)
        # if download:
        #     list(
        #         map(lambda args: download_data(root, *args),
        #             self.download_list))
        # else:
        #     list(
        #         map(lambda name, file_name, _: check_exits(root, file_name),
        #             self.download_list))

        # TODO: implelment this:

        # self.generate_image_list(root)
        if not self.has_gen_file_list:
            self.generate_image_list(root, balance_domains)
        datasets = []
        for i in range(len(CheckerboardOfficeHome211.images_lists)):
            data_list_file = os.path.join(root, CheckerboardOfficeHome211.images_lists[i])
            datasets.append(
                ImageList(
                    # TODO: Adapt the code for predicting style instead of category
                    root=root,
                    classes=self.classes(),
                    data_list_file=data_list_file,
                    transform=transforms[i],
                    **kwargs))
        self.train_dataset, self.val_dataset, self.novel_dataset = datasets
        
    def generate_image_list(
            self,
            root: str,
            balance_domains: Optional[bool] = False):
        # TODO: Produce image list if style-predicting instead of category-predicting
        train_list = ""
        val_list = ""
        novel_list = ""
        self.cat_style_matrix = torch.zeros(
            (CheckerboardOfficeHome211.num_styles,
             CheckerboardOfficeHome211.num_categories))
        styles = list(self.images_dirs.keys())

        style_indices = list(range(self.num_styles))
        if balance_domains:
            style_combs = list(combinations(style_indices, 2))
            x = ceil(self.num_categories/len(style_combs)) * len(style_combs)
            y = torch.arange(x) % len(style_combs)
            y = y.tolist()
            random.shuffle(y)
            y = y[:self.num_categories]
            bal_styles_per_cat = [style_combs[y_i] for y_i in y]
        
        for cat_index in range(self.num_categories):
            random.shuffle(style_indices)
            style_count = 0
            added_to_val = False
            for style_index in style_indices:
                style = self.images_dirs[styles[style_index]]
                cat = self.CATEGORIES[cat_index]
                image_dir = os.path.join(root,
                                         style,
                                         cat)
                for filename in os.listdir(image_dir):
                    if filename.endswith(".jpg"):
                        path = os.path.join(style, cat, filename)
                        label = self._get_label(style_index, cat_index)
                        path_and_label = path + ' ' + str(label) + '\n'
                        if balance_domains:
                            if style_index in bal_styles_per_cat[cat_index]:  
                                train_list += path_and_label
                            elif not added_to_val:
                                added_to_val = True
                                val_list += path_and_label
                                self.cat_style_matrix[style_index, cat_index] = 1
                            else:
                                novel_list += path_and_label
                                self.cat_style_matrix[style_index, cat_index] = 2
                        else:
                            if style_count < 2: 
                                train_list += path_and_label
                            elif style_count < 3:
                                train_list += path_and_label
                                self.cat_style_matrix[style_index, cat_index] = 1
                            else:
                                novel_list += path_and_label
                                self.cat_style_matrix[style_index, cat_index] = 2
                style_count += 1

        # training, validation/calibration, testing split
        train_list_filename = os.path.join(root, self.images_lists[0])
        val_list_filename = os.path.join(root, self.images_lists[1])
        novel_list_filename = os.path.join(root, self.images_lists[2])
        with open(train_list_filename, "w") as f:
            f.write(train_list)
        with open(val_list_filename, "w") as f:
            f.write(val_list)
        with open(novel_list_filename, "w") as f:
            f.write(novel_list)
        self.has_gen_file_list = True

    def __str__(self):
        str_matrix = "Categories (Cols) AND Styles (Rows) Matrix\n"
        style_index = 0
        styles = list(CheckerboardOfficeHome211.images_dirs.keys())
        # for cat_index in range(self.num_categories):
        #     str_matrix += "|" + str(cat_index)
        # str_matrix += "|\n"
        for row in self.cat_style_matrix:
            str_matrix += "|" + str(styles[style_index])
            num_train = 0
            for val in row:
                if val == 0:
                    str_matrix += "|T"
                    num_train += 1
                elif val == 1:
                    str_matrix += "|V"
                elif val == 2:
                    str_matrix += "|N"
                else:
                    str_matrix += "| "
            str_matrix += f"| Train total={num_train}\n"
            style_index += 1
        return str_matrix

    def domains(self):
        if self.style_is_domain:
            return list(CheckerboardOfficeHome211.images_dirs.keys())
        else:
            return CheckerboardOfficeHome211.CATEGORIES

    def classes(self):
        if not self.style_is_domain:
            return list(CheckerboardOfficeHome211.images_dirs.keys())
        else:
            return CheckerboardOfficeHome211.CATEGORIES

    @classmethod
    def _get_label(cls, style_index: int, category_index: int) -> int:
        return cls.num_categories * style_index + category_index

    @classmethod
    def get_category(cls, labels: torch.tensor) -> torch.tensor:
        return labels % cls.num_categories

    @classmethod
    def get_style(cls, labels: torch.tensor) -> torch.tensor:
        return labels // cls.num_categories
