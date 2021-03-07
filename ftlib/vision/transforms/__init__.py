import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
          (h, w), output size will be matched to this. If size is an int,
          output size will be (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class NormalizeAndTranspose:
    def __init__(self, mean=(104.00698793, 116.66876762, 122.67891434)):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        image = np.asarray(image, np.float32)
        # change to BGR
        image = image[:, :, ::-1]
        # normalize
        image -= self.mean
        image = image.transpose((2, 0, 1)).copy()
        return image


class DeNormalizeAndTranspose:
    def __init__(self, mean=(104.00698793, 116.66876762, 122.67891434)):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        image = image.transpose((1, 2, 0))
        # denormalize
        image += self.mean
        # change to RGB
        image = image[:, :, ::-1]
        return image


class AllApply:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return [t(image) for t in self.transforms]


class Denormalize(Normalize):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        super().__init__((-mean / std).tolist(), (1 / std).tolist())
