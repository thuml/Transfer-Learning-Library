import os
from .vision import ImageListDataset
from ._util import get_download_info


class VisDA2017(ImageListDataset):
    all_urls = {
        "train.tar": "http://csr.bu.edu/ftp/visda17/clf/train.tar",
        "validation.tar": "http://csr.bu.edu/ftp/visda17/clf/validation.tar",
        "visda2017_uda.tgz": "https://cloud.tsinghua.edu.cn/f/dab8487fe13f4dbbbf96/?dl=1"
    }
    tasks = {
        "T": {
            "data_list": "train.txt",
            "dependencies": [("train", "train.tar"), ("train.txt", "visda2017_uda.tgz")]
        },
        "V": {
            "data_list": "validation.txt",
            "dependencies": [("validation", "validation.tar"), ("validation.txt", "visda2017_uda.tgz")]
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

        super(VisDA2017, self).__init__(root, num_classes=12, data_list_file=data_list_file,
                                         download_info=download_info, **kwargs)

