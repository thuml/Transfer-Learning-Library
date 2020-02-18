import os

import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive


class VisionDataset(datasets.VisionDataset):
    """A generic data set for domain adaptation in computer vision:

    Args:
        root (string): Root directory of dataset
        num_classes (int): number of classes
        data_list_file (string): In this file, each line has two values separated by a blank space.
            The first is the relative path of an image, and the second is the label of the corresponding image.
            If your data_list_file has different formats, you need to reimplement `parse_data_file` and `__getitem__`.
        download_urls (dict: str->str) ： A dict which maps domains to download links.
            If not None, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, num_classes, data_list_file: str, download_urls=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.download_urls = download_urls

        if isinstance(download_urls, dict):
            self.download()
        
        self._num_classes = num_classes
        self.data_list = self.parse_data_file(data_list_file)
        self.loader = default_loader

    def __getitem__(self, index):
        data = self.data_list[index]
        path = data[0] if os.path.isabs(data[0]) else os.path.join(self.root, data[0])
        target = int(data[1]) if len(data) > 1 else None
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def parse_data_file(file_name):
        with open(file_name, "r") as f:
            data_list = [line.split(maxsplit=1) for line in f.readlines()]
        return data_list

    def download(self):
        """Download the data if it doesn't exist in the folder already."""
        for domain, url in self.download_urls.items():
            if self._check_exists(domain):
                continue

            print("Downloading {}".format(domain))
            download_and_extract_archive(url, download_root=self.root,
                                         filename="{}.tgz".format(domain), remove_finished=True)

    def _check_exists(self, domain):
        return os.path.exists(os.path.join(self.root, domain))

    @property
    def num_classes(self):
        return self._num_classes

