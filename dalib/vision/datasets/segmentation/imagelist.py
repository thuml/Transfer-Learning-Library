import os
from typing import List
from PIL import Image
import numpy as np
from torch.utils import data


class ImageList(data.Dataset):
    """A generic Dataset class for domain adaptation in image segmentation

    Parameters:
        - **root** (str): Root directory of dataset
        - **classes** (List[str]): The names of all the classes
        - **data_list_file** (str): File to read the image list from.
        - **transforms** (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, ``transforms.RandomCrop``.

    .. note:: In `data_list_file`, each line is the relative path of an image.
        If your data_list_file has different formats, please over-ride `parse_data_file`.
            source_dir/dog_xxx.png
            source_dir/cat_123.png
            target_dir/dog_xxy.png
            target_dir/cat_nsdf3.png
    """
    def __init__(self, root: str, classes: List[str], data_list_file: str, label_list_file: str,
                 data_folder, label_folder, mean=None, id_to_train_id=None, train_id_to_color=None, transforms=None):
        self.root = root
        self.classes = classes
        self.data_list_file = data_list_file
        self.label_list_file = label_list_file
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.mean = mean
        self.ignore_label = 255
        self.id_to_train_id = id_to_train_id
        self.train_id_to_color = train_id_to_color
        self.data_list = self.parse_data_file(self.data_list_file)
        self.label_list = self.parse_label_file(self.label_list_file)
        self.transforms = transforms

    def parse_data_file(self, data_list_file):
        with open(data_list_file, "r") as f:
            data_list = [line.strip() for line in f.readlines()]
        return data_list

    def parse_label_file(self, label_list_file):
        with open(label_list_file, "r") as f:
            label_list = [line.strip() for line in f.readlines()]
        return label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_name = self.data_list[index]
        label_name = self.label_list[index]
        image = Image.open(os.path.join(self.root, self.data_folder, image_name)).convert('RGB')
        label = Image.open(os.path.join(self.root, self.label_folder, label_name))

        image, label = self.transforms(image, label)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.int64)

        # remap label
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.int64)
        if self.id_to_train_id:
            for k, v in self.id_to_train_id.items():
                label_copy[label == k] = v

        # change to BGR
        image = image[:, :, ::-1]
        # normalize
        if self.mean is not None:
            image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy()

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    def decode_target(self, target):
        target = target.copy()
        target[target == 255] = self.num_classes
        return self.train_id_to_color[target]

    def collect_image_paths(self):
        return [os.path.join(self.root, self.data_folder, image_name) for image_name in self.data_list]