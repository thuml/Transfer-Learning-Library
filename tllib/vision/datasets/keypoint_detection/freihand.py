"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import json
import time
import torch
import os
import os.path as osp
from torchvision.datasets.utils import download_and_extract_archive

from ...transforms.keypoint_detection import *
from .keypoint_dataset import Hand21KeypointDataset
from .util import *


""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


def load_db_annotation(base_path, set_name=None):
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)

    # should have all the same length
    assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
    return list(zip(K_list, mano_list, xyz_list))


def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


""" Dataset related functions. """
def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'


class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]


    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size*cls.valid_options().index(version)


class FreiHand(Hand21KeypointDataset):
    """`FreiHand Dataset <https://lmb.informatik.uni-freiburg.de/projects/freihand/>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, ``test``, or ``all``.
        task (str, optional): The post-processing option to create dataset. Choices include ``'gs'``: green screen \
            recording, ``'auto'``: auto colorization without sample points: automatic color hallucination, \
            ``'sample'``: auto colorization with sample points, ``'hom'``: homogenized, \
            and ``'all'``: all hands. Default: 'all'.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transforms (callable, optional): A function/transform that takes in a dict (which contains PIL image and
            its labels) and returns a transformed version. E.g, :class:`~tllib.vision.transforms.keypoint_detection.Resize`.
        image_size (tuple): (width, height) of the image. Default: (256, 256)
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    .. note:: In `root`, there will exist following files after downloading.
        ::
            *.json
            training/
            evaluation/
    """
    def __init__(self, root, split='train', task='all', download=True, **kwargs):
        if download:
            if not osp.exists(osp.join(root, "training")) or not osp.exists(osp.join(root, "evaluation")):
                download_and_extract_archive("https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip",
                                             download_root=root, filename="FreiHAND_pub_v2.zip", remove_finished=False,
                                             extract_root=root)

        assert split in ['train', 'test', 'all']
        self.split = split

        assert task in ['all', 'gs', 'auto', 'sample', 'hom']
        self.task = task
        if task == 'all':
            samples = self.get_samples(root, 'gs') + self.get_samples(root, 'auto') + self.get_samples(root, 'sample') + self.get_samples(root, 'hom')
        else:
            samples = self.get_samples(root, task)
        random.seed(42)
        random.shuffle(samples)
        samples_len = len(samples)
        samples_split = min(int(samples_len * 0.2), 3200)
        if self.split == 'train':
            samples = samples[samples_split:]
        elif self.split == 'test':
            samples = samples[:samples_split]

        super(FreiHand, self).__init__(root, samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)
        keypoint3d_camera = np.array(sample['keypoint3d'])  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        # Crop the images such that the hand is at the center of the image
        # The images will be 1.5 times larger than the hand
        # The crop process will change Xc and Yc, leaving Zc with no changes
        bounding_box = get_bounding_box(keypoint2d)
        w, h = image.size
        left, upper, right, lower = scale_box(bounding_box, w, h, 1.5)
        image, keypoint2d = crop(image, upper, left, lower - upper, right - left, keypoint2d)

        # Change all hands to right hands
        if sample['left'] is False:
            image, keypoint2d = hflip(image, keypoint2d)

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
        z = keypoint3d_n[:, 2]

        meta = {
            'image': image_name,
            'keypoint2d': keypoint2d,  # （NUM_KEYPOINTS x 2）
            'keypoint3d': keypoint3d_n,  # （NUM_KEYPOINTS x 3）
            'z': z,
        }

        return image, target, target_weight, meta

    def get_samples(self, root, version='gs'):
        set = 'training'
        # load annotations of this set
        db_data_anno = load_db_annotation(root, set)

        version_map = {
            'gs': sample_version.gs,
            'hom': sample_version.hom,
            'sample': sample_version.sample,
            'auto': sample_version.auto
        }
        samples = []
        for idx in range(db_size(set)):
            image_name = os.path.join(set, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version_map[version]))
            mask_name = os.path.join(set, 'mask', '%08d.jpg' % idx)
            intrinsic_matrix, mano, keypoint3d = db_data_anno[idx]
            keypoint2d = projectPoints(keypoint3d, intrinsic_matrix)

            sample = {
                'name': image_name,
                'mask_name': mask_name,
                'keypoint2d': keypoint2d,
                'keypoint3d': keypoint3d,
                'intrinsic_matrix': intrinsic_matrix,
                'left': False
            }
            samples.append(sample)

        return samples
