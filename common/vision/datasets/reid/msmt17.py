from .basedataset import BaseImageDataset
import os.path as osp
from common.vision.datasets._util import download


class MSMT17(BaseImageDataset):
    """
    MSMT17
    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: http://www.pkuvmc.com/publications/msmt17.html
    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'msmt17'
    archive_name = 'MSMT17_V1.zip'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/55d7e5aa3c224f49b908/?dl=1'

    def __init__(self, root, verbose=True):
        super(MSMT17, self).__init__()
        download(root, self.dataset_dir, self.archive_name, self.dataset_url)
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'MSMT17_V2/mask_train_v2')
        self.test_dir = osp.join(self.dataset_dir, 'MSMT17_V2/mask_test_v2')
        self.list_train_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_gallery.txt')

        required_files = [self.dataset_dir, self.train_dir, self.test_dir]
        self.check_before_run(required_files)
        train = self.process_dir(self.train_dir, self.list_train_path)
        # val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)

        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset
