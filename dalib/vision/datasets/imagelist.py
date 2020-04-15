import os
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


class ImageList(datasets.VisionDataset):
    """A generic Dataset class for domain adaptation in image classification

    Parameters:
        - **root** (str): Root directory of dataset
        - **num_classes** (int): Number of classes
        - **data_list_file** (str): File to read the image list from.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride `parse_data_file`.
    """

    def __init__(self, root, num_classes, data_list_file, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._num_classes = num_classes
        self.data_list = self.parse_data_file(data_list_file)
        self.loader = default_loader

    def __getitem__(self, index):
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data_list[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data_list)

    def parse_data_file(self, file_name):
        """Parse file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                path, target = line.split()
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self):
        """Number of classes"""
        return self._num_classes
