import os
import random
import json
from PIL import ImageFile
import torch
from .keypoint_dataset import Body16KeypointDataset
from .._util import download as download_data, check_exits
from ...transforms.keypoint_detection import *
from .util import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Human36M(Body16KeypointDataset):
    """`Human3.6M Dataset <http://vision.imar.ro/human3.6m/description.php>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, ``test``, or ``all``.
            Default: ``train``.
        task (str, optional): Placeholder.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transforms (callable, optional): A function/transform that takes in a dict (which contains PIL image and
            its labels) and returns a transformed version. E.g, :class:`~common.vision.transforms.keypoint_detection.Resize`.
        image_size (tuple): (width, height) of the image. Default: (256, 256)
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    .. note::
        We found that the original Human3.6M image is in high resolution while most part in an image is background,
        thus we crop the image and keep only the surrounding area of hands (1.5x bigger than hands) to speed up training.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            crop_images/
            annotations/
    """
    def __init__(self, root, split='train', task='all', download=True, **kwargs):
        raise NotImplementedError("Need preprocess on Human3.6M. Not support for now.")
        assert split in ['train', 'test', 'all']
        self.split = split

        samples = []
        if self.split == 'train':
            parts = [1, 5, 6, 7, 8]
        elif self.split == 'test':
            parts = [9, 11]
        else:
            parts = [1, 5, 6, 7, 8, 9, 11]

        for part in parts:
            annotation_file = os.path.join(root, 'annotations/keypoints2d_{}.json'.format(part))
            print("loading", annotation_file)
            with open(annotation_file) as f:
                samples.extend(json.load(f))
        # decrease the number of test samples to decrease the time spent on test
        random.seed(42)
        if self.split == 'test':
            samples = random.choices(samples, k=3200)
        super(Human36M, self).__init__(root, samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = os.path.join(self.root, "crop_images", image_name)
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