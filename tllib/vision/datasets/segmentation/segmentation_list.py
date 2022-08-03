"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from typing import Sequence, Optional, Dict, Callable
from PIL import Image
import tqdm
import numpy as np
from torch.utils import data
import torch


class SegmentationList(data.Dataset):
    """A generic Dataset class for domain adaptation in image segmentation

    Args:
        root (str): Root directory of dataset
        classes (seq[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        label_list_file (str): File to read the label list from.
        data_folder (str): Sub-directory of the image.
        label_folder (str): Sub-directory of the label.
        mean (seq[float]): mean BGR value. Normalize and convert to the image if not None. Default: None.
        id_to_train_id (dict, optional): the map between the id on the label and the actual train id.
        train_id_to_color (seq, optional): the map between the train id and the color.
        transforms (callable, optional): A function/transform that  takes in  (PIL Image, label) pair \
            and returns a transformed version. E.g, :class:`~tllib.vision.transforms.segmentation.Resize`.

    .. note:: In ``data_list_file``, each line is the relative path of an image.
        If your data_list_file has different formats, please over-ride :meth:`~SegmentationList.parse_data_file`.
        ::
            source_dir/dog_xxx.png
            target_dir/dog_xxy.png

        In ``label_list_file``, each line is the relative path of an label.
        If your label_list_file has different formats, please over-ride :meth:`~SegmentationList.parse_label_file`.

    .. warning:: When mean is not None, please do not provide Normalize and ToTensor in transforms.

    """
    def __init__(self, root: str, classes: Sequence[str], data_list_file: str, label_list_file: str,
                 data_folder: str, label_folder: str,
                 id_to_train_id: Optional[Dict] = None, train_id_to_color: Optional[Sequence] = None,
                 transforms: Optional[Callable] = None):
        self.root = root
        self.classes = classes
        self.data_list_file = data_list_file
        self.label_list_file = label_list_file
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.ignore_label = 255
        self.id_to_train_id = id_to_train_id
        self.train_id_to_color = np.array(train_id_to_color)
        self.data_list = self.parse_data_file(self.data_list_file)
        self.label_list = self.parse_label_file(self.label_list_file)
        self.transforms = transforms

    def parse_data_file(self, file_name):
        """Parse file to image list

        Args:
            file_name (str): The path of data file

        Returns:
            List of image path
        """
        with open(file_name, "r") as f:
            data_list = [line.strip() for line in f.readlines()]
        return data_list

    def parse_label_file(self, file_name):
        """Parse file to label list

        Args:
            file_name (str): The path of data file

        Returns:
            List of label path
        """
        with open(file_name, "r") as f:
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

        # remap label
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        label = np.asarray(label, np.int64)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.int64)
        if self.id_to_train_id:
            for k, v in self.id_to_train_id.items():
                label_copy[label == k] = v

        return image, label_copy.copy()

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    def decode_target(self, target):
        """ Decode label (each value is integer) into the corresponding RGB value.

        Args:
            target (numpy.array): label in shape H x W

        Returns:
            RGB label (PIL Image) in shape H x W x 3
        """
        target = target.copy()
        target[target == 255] = self.num_classes # unknown label is black on the RGB label
        target = self.train_id_to_color[target]
        return Image.fromarray(target.astype(np.uint8))

    def collect_image_paths(self):
        """Return a list of the absolute path of all the images"""
        return [os.path.join(self.root, self.data_folder, image_name) for image_name in self.data_list]

    @staticmethod
    def _save_pil_image(image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)

    def translate(self, transform: Callable, target_root: str, color=False):
        """ Translate an image and save it into a specified directory

        Args:
            transform (callable): a transform function that maps (image, label) pair from one domain to another domain
            target_root (str): the root directory to save images and labels

        """
        os.makedirs(target_root, exist_ok=True)
        for image_name, label_name in zip(tqdm.tqdm(self.data_list), self.label_list):
            image_path = os.path.join(target_root, self.data_folder, image_name)
            label_path = os.path.join(target_root, self.label_folder, label_name)
            if os.path.exists(image_path) and os.path.exists(label_path):
                continue
            image = Image.open(os.path.join(self.root, self.data_folder, image_name)).convert('RGB')
            label = Image.open(os.path.join(self.root, self.label_folder, label_name))

            translated_image, translated_label = transform(image, label)
            self._save_pil_image(translated_image, image_path)
            self._save_pil_image(translated_label, label_path)
            if color:
                colored_label = self.decode_target(np.array(translated_label))
                file_name, file_ext = os.path.splitext(label_name)
                self._save_pil_image(colored_label, os.path.join(target_root, self.label_folder,
                                                                 "{}_color{}".format(file_name, file_ext)))

    @property
    def evaluate_classes(self):
        """The name of classes to be evaluated"""
        return self.classes

    @property
    def ignore_classes(self):
        """The name of classes to be ignored"""
        return list(set(self.classes) - set(self.evaluate_classes))