"""
Modified from https://github.com/SikaStar/IDM
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from .basedataset import BaseImageDataset
from typing import Callable
import os.path as osp
from common.vision.datasets._util import download


class UnrealPerson(BaseImageDataset):
    """UnrealPerson dataset from `UnrealPerson: An Adaptive Pipeline towards Costless Person Re-identification
    (CVPR 2021) <https://arxiv.org/pdf/2012.04268v2.pdf>`_.

    Dataset statistics:
        - identities: 3000
        - images: 120,000
        - cameras: 34

    Args:
        root (str): Root directory of dataset
        verbose (bool, optional): If true, print dataset statistics after loading the dataset. Default: True
    """
    dataset_dir = '.'
    download_list = [
        ("list_unreal_train.txt", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/a51b22fd760743e7bca6/?dl=1"),
        ("unreal_v1.1", "unreal_v1.1.tar", "https://cloud.tsinghua.edu.cn/f/a8806bb3bf1744dda5b1/?dl=1"),
        ("unreal_v1.2", "unreal_v1.2.tar", "https://cloud.tsinghua.edu.cn/f/449224485a654c5baa8f/?dl=1"),
        ("unreal_v1.3", "unreal_v1.3.tar", "https://cloud.tsinghua.edu.cn/f/069f3162f74849c09c10/?dl=1"),
        ("unreal_v2.1", "unreal_v2.1.tar", "https://cloud.tsinghua.edu.cn/f/a791aaa42674466eb183/?dl=1"),
        ("unreal_v2.2", "unreal_v2.2.tar", "https://cloud.tsinghua.edu.cn/f/b601d9f54f964248bd0e/?dl=1"),
        ("unreal_v2.3", "unreal_v2.3.tar", "https://cloud.tsinghua.edu.cn/f/311ec60e810b42d48d12/?dl=1"),
        ("unreal_v3.1", "unreal_v3.1.tar", "https://cloud.tsinghua.edu.cn/f/d51b7c1d125e4632bcf9/?dl=1"),
        ("unreal_v3.2", "unreal_v3.2.tar", "https://cloud.tsinghua.edu.cn/f/4efbd969ea2f4e8197e8/?dl=1"),
        ("unreal_v3.3", "unreal_v3.3.tar", "https://cloud.tsinghua.edu.cn/f/a3cc3d9c460247848fb7/?dl=1"),
        ("unreal_v4.1", "unreal_v4.1.tar", "https://cloud.tsinghua.edu.cn/f/ca05183ac9cd4be5a53b/?dl=1"),
        ("unreal_v4.2", "unreal_v4.2.tar", "https://cloud.tsinghua.edu.cn/f/b90722cbd754496f9f40/?dl=1"),
        ("unreal_v4.3", "unreal_v4.3.tar", "https://cloud.tsinghua.edu.cn/f/547ae646c3d346038297/?dl=1"),
    ]

    def __init__(self, root, verbose=True):
        super(UnrealPerson, self).__init__()
        list(map(lambda args: download(root, *args), self.download_list))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_list = osp.join(self.dataset_dir, 'list_unreal_train.txt')

        required_files = [self.dataset_dir, self.train_list]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list)
        self.train = train
        self.query = []
        self.gallery = []
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        if verbose:
            print("=> UnrealPerson loaded")
            print("  ----------------------------------------")
            print("  subset   | # ids | # cams | # images")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:5d} | {:8d}"
                  .format(self.num_train_pids, self.num_train_cams, self.num_train_imgs))
            print("  ----------------------------------------")

    def process_dir(self, list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        dataset = []
        pid_container = set()
        for line in lines:
            line = line.strip()
            pid = line.split(' ')[1]
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        for line in lines:
            line = line.strip()
            fname, pid, cid = line.split(' ')[0], line.split(' ')[1], int(line.split(' ')[2])
            img_path = osp.join(self.dataset_dir, fname)
            dataset.append((img_path, pid2label[pid], cid))

        return dataset

    def translate(self, transform: Callable, target_root: str):
        raise NotImplementedError
