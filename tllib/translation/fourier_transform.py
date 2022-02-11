"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import numpy as np
import os
import tqdm
import random
from PIL import Image
from typing import Optional, Sequence
import torch.nn as nn


def low_freq_mutate(amp_src: np.ndarray, amp_trg: np.ndarray, beta: Optional[int] = 1):
    """
    Args:
        amp_src (numpy.ndarray): amplitude component of the Fourier transform of source image
        amp_trg (numpy.ndarray): amplitude component of the Fourier transform of target image
        beta (int, optional): the size of the center region to be replace. Default: 1

    Returns:
        amplitude component of the Fourier transform of source image
        whose low-frequency component is replaced by that of the target image.

    """
    # Shift the zero-frequency component to the center of the spectrum.
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    # The low-frequency component includes
    # the area where the horizontal and vertical distance from the center does not exceed beta
    _, h, w = a_src.shape
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - beta
    h2 = c_h + beta + 1
    w1 = c_w - beta
    w2 = c_w + beta + 1

    # The low-frequency component of source amplitude is replaced by the target amplitude
    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


class FourierTransform(nn.Module):
    """
    Fourier Transform is introduced by `FDA: Fourier Domain Adaptation for Semantic Segmentation (CVPR 2020) <https://arxiv.org/abs/2004.05498>`_

    Fourier Transform replace the low frequency component of the amplitude of the source image to that of the target image.
    Denote with :math:`M_{β}` a mask, whose value is zero except for the center region:

    .. math::
        M_{β}(h,w) = \mathbb{1}_{(h, w)\in [-β,β, -β, β]}

    Given images :math:`x^s` from source domain and :math:`x^t` from target domain, the source image in the target style is

    .. math::
        x^{s→t} = \mathcal{F}^{-1}([ M_{β}\circ\mathcal{F}^A(x^t) + (1-M_{β})\circ\mathcal{F}^A(x^s), \mathcal{F}^P(x^s) ])

    where :math:`\mathcal{F}^A`, :math:`\mathcal{F}^P` are the amplitude and phase component of the Fourier
    Transform :math:`\mathcal{F}` of an RGB image.

    Args:
        image_list (sequence[str]): A sequence of image list from the target domain.
        amplitude_dir (str): Specifies the directory to put the amplitude component of the target image.
        beta (int, optional): :math:`β`. Default: 1.
        rebuild (bool, optional): whether rebuild the amplitude component of the target image in the given directory.

    Inputs:
        - image (PIL Image): image from the source domain, :math:`x^t`.

    Examples:

        >>> from tllib.translation.fourier_transform import FourierTransform
        >>> image_list = ["target_image_path1", "target_image_path2"]
        >>> amplitude_dir = "path/to/amplitude_dir"
        >>> fourier_transform = FourierTransform(image_list, amplitude_dir, beta=1, rebuild=False)
        >>> source_image = np.array((256, 256, 3)) # image form source domain
        >>> source_image_in_target_style = fourier_transform(source_image)

    .. note::
        The meaning of :math:`β` is different from that of the origin paper. Experimentally, we found that the size of
        the center region in the frequency space should be constant when the image size increases. Thus we make the size
        of the center region independent of the image size. A recommended value for :math:`β` is 1.

    .. note::
        The image structure of the source domain and target domain should be as similar as possible,
        thus for segemntation tasks, FourierTransform should be used before RandomResizeCrop and other transformations.

    .. note::
        The image size of the source domain and the target domain need to be the same, thus before FourierTransform,
        you should use Resize to convert the source image to the target image size.

    Examples:

        >>> from tllib.translation.fourier_transform import FourierTransform
        >>> import tllibvision.datasets.segmentation.transforms as T
        >>> from PIL import Image
        >>> target_image_list = ["target_image_path1", "target_image_path2"]
        >>> amplitude_dir = "path/to/amplitude_dir"
        >>> # build a fourier transform that translate source images to the target style
        >>> fourier_transform = T.wrapper(FourierTransform)(target_image_list, amplitude_dir)
        >>> transforms=T.Compose([
        ...     # convert source image to the size of the target image before fourier transform
        ...     T.Resize((2048, 1024)),
        ...     fourier_transform,
        ...     T.RandomResizedCrop((1024, 512)),
        ...     T.RandomHorizontalFlip(),
        ... ])
        >>> source_image = Image.open("path/to/source_image") # image form source domain
        >>> source_image_in_target_style = transforms(source_image)
    """
    # TODO add image examples when beta is different
    def __init__(self, image_list: Sequence[str], amplitude_dir: str,
                 beta: Optional[int] = 1, rebuild: Optional[bool] = False):
        super(FourierTransform, self).__init__()
        self.amplitude_dir = amplitude_dir
        if not os.path.exists(amplitude_dir) or rebuild:
            os.makedirs(amplitude_dir, exist_ok=True)
            self.build_amplitude(image_list, amplitude_dir)
        self.beta = beta
        self.length = len(image_list)

    @staticmethod
    def build_amplitude(image_list, amplitude_dir):
        # extract amplitudes from target domain
        for i, image_name in enumerate(tqdm.tqdm(image_list)):
            image = Image.open(image_name).convert('RGB')
            image = np.asarray(image, np.float32)
            image = image.transpose((2, 0, 1))
            fft = np.fft.fft2(image, axes=(-2, -1))
            amp = np.abs(fft)
            np.save(os.path.join(amplitude_dir, "{}.npy".format(i)), amp)

    def forward(self, image):
        # randomly sample a target image and load its amplitude component
        amp_trg = np.load(os.path.join(self.amplitude_dir, "{}.npy".format(random.randint(0, self.length-1))))

        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))

        # get fft, amplitude on source domain
        fft_src = np.fft.fft2(image, axes=(-2, -1))
        amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
        # mutate the amplitude part of source with target
        amp_src_ = low_freq_mutate(amp_src, amp_trg, beta=self.beta)

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp(1j * pha_src)

        # get the mutated image
        src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
        src_in_trg = np.real(src_in_trg)

        src_in_trg = src_in_trg.transpose((1, 2, 0))
        src_in_trg = Image.fromarray(src_in_trg.clip(min=0, max=255).astype('uint8')).convert('RGB')

        return src_in_trg
