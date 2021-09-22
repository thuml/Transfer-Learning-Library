"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import scipy.io as scio
import os

from PIL import ImageFile
import torch
from .keypoint_dataset import Body16KeypointDataset
from ...transforms.keypoint_detection import *
from .util import *
from .._util import download as download_data, check_exits


ImageFile.LOAD_TRUNCATED_IMAGES = True


class LSP(Body16KeypointDataset):
    """`Leeds Sports Pose Dataset <http://sam.johnson.io/research/lsp.html>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): PlaceHolder.
        task (str, optional): Placeholder.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transforms (callable, optional): PlaceHolder.
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    .. note:: In `root`, there will exist following files after downloading.
        ::
            lsp/
                images/
                joints.mat

    .. note::
        LSP is only used for target domain. Due to the small dataset size, the whole dataset is used
        no matter what ``split`` is. Also, the transform is fixed.
    """
    def __init__(self, root, split='train', task='all', download=True, image_size=(256, 256), transforms=None, **kwargs):
        if download:
            download_data(root, "images", "lsp_dataset.zip",
                          "https://cloud.tsinghua.edu.cn/f/46ea73c89abc46bfb125/?dl=1")
        else:
            check_exits(root, "lsp")

        assert split in ['train', 'test', 'all']
        self.split = split

        samples = []
        annotations = scio.loadmat(os.path.join(root, "joints.mat"))['joints'].transpose((2, 1, 0))
        for i in range(0, 2000):
            image = "im{0:04d}.jpg".format(i+1)
            annotation = annotations[i]
            samples.append((image, annotation))

        self.joints_index = (0, 1, 2, 3, 4, 5, 13, 13, 12, 13, 6, 7, 8, 9, 10, 11)
        self.visible = np.array([1.] * 6 + [0, 0] + [1.] * 8, dtype=np.float32)
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms = Compose([
            ResizePad(image_size[0]),
            ToTensor(),
            normalize
        ])
        super(LSP, self).__init__(root, samples, transforms=transforms, image_size=image_size, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample[0]
        image = Image.open(os.path.join(self.root, "images", image_name))
        keypoint2d = sample[1][self.joints_index, :2]
        image, data = self.transforms(image, keypoint2d=keypoint2d)
        keypoint2d = data['keypoint2d']
        visible = self.visible * (1-sample[1][self.joints_index, 2])
        visible = visible[:, np.newaxis]

        # 2D heatmap
        target, target_weight = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_name,
            'keypoint2d': keypoint2d,  # （NUM_KEYPOINTS x 2）
            'keypoint3d': np.zeros((self.num_keypoints, 3)).astype(keypoint2d.dtype),  # （NUM_KEYPOINTS x 3）
        }
        return image, target, target_weight, meta
