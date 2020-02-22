import os
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive


class ImageListDataset(datasets.VisionDataset):
    """A generic data set for domain adaptation in image classification:

    Args:
        root (string): Root directory of dataset
        num_classes (int): number of classes
        data_list_file (string): In this file, each line has two values separated by a blank space.
            The first is the relative path of an image, and the second is the label of the corresponding image.
            If your data_list_file has different formats, you need to reimplement `parse_data_file` and `__getitem__`.
        download_info (list): Download information. Each value is a triple (file_name, archive_name, url_link).
            If not None, downloads the dataset from the internet and puts it in root directory.
            If `file_name` already exists, it is not downloaded again.
            Else `archive_name` will be downloaded from `url_link` and extracted to `file_name`.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, num_classes, data_list_file: str, download_info=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.download_info = download_info

        if isinstance(download_info, list):
            self.download()
        
        self._num_classes = num_classes
        self.data_list = self.parse_data_file(data_list_file)
        self.loader = default_loader

    def __getitem__(self, index):
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
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                path, target = line.split()
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    def download(self):
        """Download the data if it doesn't exist in the folder already."""
        for file_name, archive_name, url_link in self.download_info:
            if self._check_exists(file_name):
                continue

            print("Downloading {}".format(file_name))
            download_and_extract_archive(url_link, download_root=self.root,
                                         filename=archive_name, remove_finished=True)

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self.root, filename))

    @property
    def num_classes(self):
        return self._num_classes

