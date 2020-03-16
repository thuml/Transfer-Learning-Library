import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OfficeHome(ImageList):
    """`OfficeHome <http://hemanthdv.org/OfficeHome-Dataset/>`_ Dataset.

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
            ``'Cl'``: Clipart, ``'Pr'``: Product and ``'Rw'``: Real_World.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/ee615d5ad5e146278a80/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/81a4f30c7e894298b435/?dl=1"),
        ("Clipart", "Clipart.tgz", "https://cloud.tsinghua.edu.cn/f/d4ad15137c734917aa5c/?dl=1"),
        ("Product", "Product.tgz", "https://cloud.tsinghua.edu.cn/f/a6b643999c574184bbcd/?dl=1"),
        ("Real_World", "Real_World.tgz", "https://cloud.tsinghua.edu.cn/f/60ca8452bcf743408245/?dl=1")
    ]
    image_list = {
        "Ar": "image_list/Art.txt",
        "Cl": "image_list/Clipart.txt",
        "Pr": "image_list/Product.txt",
        "Rw": "image_list/Real_World.txt",
    }

    def __init__(self, root, task, download=False, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(OfficeHome, self).__init__(root, num_classes=65, data_list_file=data_list_file, **kwargs)

