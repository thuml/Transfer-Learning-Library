import os
from .vision import VisionDataset
from ._util import subset


class Office31(VisionDataset):
    all_urls = {
        "amazon": "https://cloud.tsinghua.edu.cn/f/05640442cd904c39ad60/?dl=1",
        "dslr": "https://cloud.tsinghua.edu.cn/f/a069d889628d4b468c32/?dl=1",
        "webcam": "https://cloud.tsinghua.edu.cn/f/4c4afebf51384cf1aa95/?dl=1"}
    tasks = {
        "A": {
            "file": "amazon.txt",
            "domains": ["amazon"]
        },
        "D": {
            "file": "dslr.txt",
            "domains": ["dslr"]
        },
        "W": {
            "file": "webcam.txt",
            "domains": ["webcam"]
        }
    }

    def __init__(self, root, task, download=False, **kwargs):
        assert task in self.tasks
        data_list_file = os.path.join(root, self.tasks[task]["file"])
        # Download only the data needed for this task.
        if download:
            download_urls = subset(self.all_urls, self.tasks[task]['domains'])
        else:
            download_urls = None

        super(Office31, self).__init__(root, data_list_file=data_list_file,
                                       download_urls=download_urls,  **kwargs)


