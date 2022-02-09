"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from PIL import Image
import random
import math
from typing import ClassVar, Sequence, List, Tuple
from torch import Tensor
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
import torch.nn as nn
from . import MultipleApply as MultipleApplyBase, NormalizeAndTranspose as NormalizeAndTransposeBase


def wrapper(transform: ClassVar):
    """ Wrap a transform for classification to a transform for segmentation.
    Note that the segmentation label will keep the same before and after wrapper.

    Args:
        transform (class, callable): transform for classification

    Returns:
        transform for segmentation
    """
    class WrapperTransform(transform):
        def __call__(self, image, label):
            image = super().__call__(image)
            return image, label
    return WrapperTransform


ColorJitter = wrapper(T.ColorJitter)
Normalize = wrapper(T.Normalize)
ToTensor = wrapper(T.ToTensor)
ToPILImage = wrapper(T.ToPILImage)
MultipleApply = wrapper(MultipleApplyBase)
NormalizeAndTranspose = wrapper(NormalizeAndTransposeBase)


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Resize((512, 512)),
        >>>     RandomHorizontalFlip()
        >>> ])
    """
    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(nn.Module):
    """Resize the input image and the corresponding label to the given size.
    The image should be a PIL Image.

    Args:
        image_size (sequence): The requested image size in pixels, as a 2-tuple:
          (width, height).
        label_size (sequence, optional): The requested segmentation label size in pixels, as a 2-tuple:
          (width, height). The same as image_size if None. Default: None.
    """

    def __init__(self, image_size, label_size=None):
        super(Resize, self).__init__()
        self.image_size = image_size
        if label_size is None:
            self.label_size = image_size
        else:
            self.label_size = label_size

    def forward(self, image, label):
        """
        Args:
            image: (PIL Image): Image to be scaled.
            label: (PIL Image): Segmentation label to be scaled.

        Returns:
            Rescaled image, rescaled segmentation label
        """
        # resize
        image = image.resize(self.image_size, Image.BICUBIC)
        label = label.resize(self.label_size, Image.NEAREST)
        return image, label


class RandomCrop(nn.Module):
    """Crop the given image at a random location.
    The image can be a PIL Image

    Args:
        size (sequence): Desired output size of the crop.
    """
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, image, label):
        """
        Args:
            image: (PIL Image): Image to be cropped.
            label: (PIL Image): Segmentation label to be cropped.

        Returns:
            Cropped image, cropped segmentation label.
        """
        # random crop
        left = image.size[0] - self.size[0]
        upper = image.size[1] - self.size[1]

        left = random.randint(0, left-1)
        upper = random.randint(0, upper-1)
        right = left + self.size[0]
        lower = upper + self.size[1]

        image = image.crop((left, upper, right, lower))
        label = label.crop((left, upper, right, lower))
        return image, label


class RandomHorizontalFlip(nn.Module):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p

    def forward(self, image, label):
        """
        Args:
            image: (PIL Image): Image to be flipped.
            label: (PIL Image): Segmentation label to be flipped.

        Returns:
            Randomly flipped image, randomly flipped segmentation label.
        """
        if random.random() < self.p:
            return F.hflip(image), F.hflip(label)
        return image, label


class RandomResizedCrop(T.RandomResizedCrop):
    """Crop the given image to random size and aspect ratio.
    The image can be a PIL Image.

    A crop of random size (default: of 0.5 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.

    Args:
        size (int or sequence): expected output size of each edge. If size is an
          int instead of sequence like (h, w), a square output size ``(size, size)`` is
          made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple of float): range of size of the origin size cropped
        ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BICUBIC):
        super(RandomResizedCrop, self).__init__(size, scale, ratio, interpolation)

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            params (i, j, h, w) to be passed to ``crop`` for a random sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = area * random.uniform(scale[0], scale[1])
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = math.exp(random.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, image, label):
        """
        Args:
            image: (PIL Image): Image to be cropped and resized.
            label: (PIL Image): Segmentation label to be cropped and resized.

        Returns:
            Randomly cropped and resized image, randomly cropped and resized segmentation label.
        """
        top, left, height, width = self.get_params(image, self.scale, self.ratio)
        image = image.crop((left, top, left + width, top + height))
        image = image.resize(self.size, self.interpolation)
        label = label.crop((left, top, left + width, top + height))
        label = label.resize(self.size, Image.NEAREST)
        return image, label


class RandomChoice(T.RandomTransforms):
    """Apply single transformation randomly picked from a list.
    """
    def __call__(self, image, label):
        t = random.choice(self.transforms)
        return t(image, label)


class RandomApply(T.RandomTransforms):
    """Apply randomly a list of transformations with a given probability.

    Args:
        transforms (list or tuple or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, image, label):
        if self.p < random.random():
            return image
        for t in self.transforms:
            image, label = t(image, label)
        return image
