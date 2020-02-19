import os
from .vision import ImageListDataset
from ._util import get_download_info


class Office31(ImageListDataset):
    all_urls = {
        "amazon.tgz": "https://cloud.tsinghua.edu.cn/f/05640442cd904c39ad60/?dl=1",
        "dslr.tgz": "https://cloud.tsinghua.edu.cn/f/a069d889628d4b468c32/?dl=1",
        "webcam.tgz": "https://cloud.tsinghua.edu.cn/f/4c4afebf51384cf1aa95/?dl=1",
        "office31_uda.tgz": "https://cloud.tsinghua.edu.cn/f/d72790c8d75c422ab393/?dl=1",
    }
    tasks = {
        "A": {
            "data_list": "amazon.txt",
            "dependencies": [("amazon", "amazon.tgz"), ("amazon.txt", "office31_uda.tgz")]
        },
        "D": {
            "data_list": "dslr.txt",
            "dependencies": [("dslr", "dslr.tgz"), ("dslr.txt", "office31_uda.tgz")]
        },
        "W": {
            "data_list": "webcam.txt",
            "dependencies": [("webcam", "webcam.tgz"), ("webcam.txt", "office31_uda.tgz")]
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

        super(Office31, self).__init__(root, num_classes=31, data_list_file=data_list_file,
                                       download_info=download_info,  **kwargs)

