from .basedataset import BaseImageDataset
import os
import os.path as osp
from common.vision.datasets._util import download


class MSMT17(BaseImageDataset):
    """MSMT17 dataset from `Person Transfer GAN to Bridge Domain Gap for Person Re-Identification (CVPR 2018)
    <https://arxiv.org/pdf/1711.08565.pdf>`_.

    Dataset statistics:
        - identities: 4101
        - images: 32621 (train) + 11659 (query) + 82161 (gallery)
        - cameras: 15

    Args:
        root (str): Root directory of dataset
        verbose (bool, optional): If true, print dataset statistics after loading the dataset. Default: True
    """
    dataset_dir = '.'
    archive_name = 'MSMT17_V1.zip'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/55d7e5aa3c224f49b908/?dl=1'

    def __init__(self, root, verbose=True):
        super(MSMT17, self).__init__()
        download(root, self.dataset_dir, self.archive_name, self.dataset_url)
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        self.train = self.process_dir(self.train_dir)
        self.query = self.process_dir(self.query_dir)
        self.gallery = self.process_dir(self.gallery_dir)
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def process_dir(self, dir_path):
        image_list = os.listdir(dir_path)
        dataset = []
        pid_container = set()

        for image_path in image_list:
            pid, camid, _ = image_path.split('_')
            pid = int(pid)
            # camera id of MSMT starts from 1, so we subtract 1 here
            camid = int(camid[1:]) - 1
            full_image_path = osp.join(dir_path, image_path)
            dataset.append((full_image_path, pid, camid))
            pid_container.add(pid)

        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset
