"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
import json
import random
from PIL import ImageFile, Image
import torch
import os.path as osp

from .._util import download as download_data, check_exits
from .keypoint_dataset import Hand21KeypointDataset
from .util import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Hand3DStudio(Hand21KeypointDataset):
    """`Hand-3d-Studio Dataset <https://www.yangangwang.com/papers/ZHAO-H3S-2020-02.html>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, ``test``, or ``all``.
        task (str, optional): The task to create dataset. Choices include ``'noobject'``: only hands without objects, \
            ``'object'``: only hands interacting with hands, and ``'all'``: all hands. Default: 'noobject'.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transforms (callable, optional): A function/transform that takes in a dict (which contains PIL image and
            its labels) and returns a transformed version. E.g, :class:`~common.vision.transforms.keypoint_detection.Resize`.
        image_size (tuple): (width, height) of the image. Default: (256, 256)
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    .. note::
        We found that the original H3D image is in high resolution while most part in an image is background,
        thus we crop the image and keep only the surrounding area of hands (1.5x bigger than hands) to speed up training.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            H3D_crop/
                annotation.json
                part1/
                part2/
                part3/
                part4/
                part5/
    """
    def __init__(self, root, split='train', task='noobject', download=True, **kwargs):
        assert split in ['train', 'test', 'all']
        self.split = split
        assert task in ['noobject', 'object', 'all']
        self.task = task

        if download:
            download_data(root, "H3D_crop", "H3D_crop.tar", "https://cloud.tsinghua.edu.cn/f/d4e612e44dc04d8eb01f/?dl=1")
        else:
            check_exits(root, "H3D_crop")

        root = osp.join(root, "H3D_crop")
        # load labels
        annotation_file = os.path.join(root, 'annotation.json')
        print("loading from {}".format(annotation_file))
        with open(annotation_file) as f:
            samples = list(json.load(f))
        if task == 'noobject':
            samples = [sample for sample in samples if int(sample['without_object']) == 1]
        elif task == 'object':
            samples = [sample for sample in samples if int(sample['without_object']) == 0]

        random.seed(42)
        random.shuffle(samples)
        samples_len = len(samples)
        samples_split = min(int(samples_len * 0.2), 3200)
        if split == 'train':
            samples = samples[samples_split:]
        elif split == 'test':
            samples = samples[:samples_split]

        super(Hand3DStudio, self).__init__(root, samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)
        keypoint3d_camera = np.array(sample['keypoint3d'])  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        image, data = self.transforms(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d = data['keypoint2d']
        intrinsic_matrix = data['intrinsic_matrix']
        keypoint3d_camera = keypoint2d_to_3d(keypoint2d, intrinsic_matrix, Zc)

        # noramlize 2D pose:
        visible = np.ones((self.num_keypoints, ), dtype=np.float32)
        visible = visible[:, np.newaxis]
        # 2D heatmap
        target, target_weight = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        # normalize 3D pose:
        # put middle finger metacarpophalangeal (MCP) joint in the center of the coordinate system
        # and make distance between wrist and middle finger MCP joint to be of length 1
        keypoint3d_n = keypoint3d_camera - keypoint3d_camera[9:10, :]
        keypoint3d_n = keypoint3d_n / np.sqrt(np.sum(keypoint3d_n[0, :] ** 2))

        meta = {
            'image': image_name,
            'keypoint2d': keypoint2d,  # （NUM_KEYPOINTS x 2）
            'keypoint3d': keypoint3d_n,  # （NUM_KEYPOINTS x 3）
        }
        return image, target, target_weight, meta


class Hand3DStudioAll(Hand3DStudio):
    """
    `Hand-3d-Studio Dataset <https://www.yangangwang.com/papers/ZHAO-H3S-2020-02.html>`_

    """
    def __init__(self,  root, task='all', **kwargs):
        super(Hand3DStudioAll, self).__init__(root, task=task, **kwargs)