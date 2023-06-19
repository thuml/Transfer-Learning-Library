from torch.utils.data.dataset import Dataset

import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
import json


class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """

    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal=None):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear',
                             align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(
            0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        if normal is not None:
            normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear',
                                align_corners=True).squeeze(0)
            return img_, label_, depth_ / sc, normal_
        else:
            return img_, label_, depth_ / sc


class NYUv2(Dataset):
    """
    We could further improve the performance with the data augmentation of NYUv2 defined in:
        [1] PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing
        [2] Pattern affinitive propagation across depth, surface normal and semantic segmentation
        [3] Mti-net: Multiscale task interaction networks for multi-task learning

        1. Random scale in a selected raio 1.0, 1.2, and 1.5.
        2. Random horizontal flip.

    Please note that: all baselines and MTAN did NOT apply data augmentation in the original paper.
    """

    def __init__(self, root, mode='train', augmentation=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.image_size = (288, 384)

        with open(os.path.join(root, 'data_split.json'), 'r') as f:
            data_split = json.load(f)
        train_index, val_index = data_split['train'], data_split['val']
        # read the data file
        if self.mode == 'train':
            self.index_list = train_index
            self.data_path = self.root + '/train'
        elif self.mode == 'val':
            self.index_list = val_index
            self.data_path = self.root + '/train'
        elif self.mode == 'trainval':
            self.index_list = train_index + val_index
            self.data_path = self.root + '/train'
        elif self.mode == 'test':
            data_len = len(fnmatch.filter(os.listdir(self.root + '/val/image'), '*.npy'))
            self.index_list = list(range(data_len))
            self.data_path = self.root + '/val'

        # calculate data length
        self.noise = torch.rand(len(self.index_list), 1, 288, 384)
        self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3, 'noise': 1}
        self.num_classes = 13

    def __getitem__(self, i):
        index = self.index_list[i]
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))
        noise = self.noise[i]

        # apply data augmentation if required
        if self.augmentation:
            image, semantic, depth, normal = RandomScaleCrop()(image, semantic, depth, normal)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
                normal = torch.flip(normal, dims=[2])
                normal[0, :, :] = - normal[0, :, :]

        # hack no augmentation for noise
        return image.float(), {'segmentation': semantic.float(), 'depth': depth.float(), 'normal': normal.float(),
                               'noise': noise.float()}

    def __len__(self):
        return len(self.index_list)
