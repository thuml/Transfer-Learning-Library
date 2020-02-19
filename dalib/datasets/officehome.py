import os
from .vision import ImageListDataset
from ._util import get_download_info


class OfficeHome(ImageListDataset):
    all_urls = {
        "Art.tgz": "https://cloud.tsinghua.edu.cn/f/81a4f30c7e894298b435/?dl=1",
        "Clipart.tgz": "https://cloud.tsinghua.edu.cn/f/d4ad15137c734917aa5c/?dl=1",
        "Product.tgz": "https://cloud.tsinghua.edu.cn/f/a6b643999c574184bbcd/?dl=1",
        "Real_World.tgz": "https://cloud.tsinghua.edu.cn/f/60ca8452bcf743408245/?dl=1",
        "office_home_uda.tgz": "https://cloud.tsinghua.edu.cn/f/3239ce6c52c1465bb02f/?dl=1",
    }
    tasks = {
        "Ar": {
            "data_list": "Art.txt",
            "dependencies": [("Art", "Art.tgz"), ("Art.txt", "office_home_uda.tgz")]
        },
        "Cl": {
            "data_list": "Clipart.txt",
            "dependencies": [("Clipart", "Clipart.tgz"), ("Clipart.txt", "office_home_uda.tgz")]
        },
        "Pr": {
            "data_list": "Product.txt",
            "dependencies": [("Product", "Product.tgz"), ("Product.txt", "office_home_uda.tgz")]
        },
        "Rw": {
            "data_list": "Real_World.txt",
            "dependencies": [("Real_World", "Real_World.tgz"), ("Real_World.txt", "office_home_uda.tgz")]
        }
    }

    def __init__(self, root, task, download=False, **kwargs):
        assert task in self.tasks
        data_list_file = os.path.join(root, self.tasks[task]["data_list"])
        # Download only the data needed for this task.
        if download:
            download_info = get_download_info(self.all_urls, self.tasks[task]['dependencies'])
        else:
            download_info = None

        super(OfficeHome, self).__init__(root, num_classes=65, data_list_file=data_list_file,
                                       download_info=download_info,  **kwargs)

