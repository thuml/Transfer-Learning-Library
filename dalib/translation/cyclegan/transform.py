import torch
import torch.nn as nn
import torchvision.transforms as T
from ...vision.transforms import Denormalize


class Translation(nn.Module):
    """
    Image Translation Transform Module

    Args:
        generator (torch.nn.Module): An image generator, e.g. :meth:`~dalib.translation.cyclegan.resnet_9_generator`
        device (torch.device): device to put the generator. Default: 'cpu'
        pre_process (callable): the transform before the image is fed to the generator. Default: None
        post_process (callable): the transform after the image is fed to the generator. Default: None

    Input:
        - image (PIL.Image): raw image in shape H x W x C

    Output:
        raw image in shape H x W x 3

    .. note::
        - When ``pre_process`` is None, the image will be converted into tensor and normalized.
        - When ``post_process`` is None, the image will be denormalized and converted into PIL.Image

    """
    def __init__(self, generator, device=torch.device("cpu"), pre_process=None, post_process=None):
        super(Translation, self).__init__()
        # device = torch.device("cpu")
        self.generator = generator.to(device)
        self.device = device
        if pre_process is None:
            pre_process = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        if post_process is None:
            post_process = T.Compose([
                Denormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                T.ToPILImage()
            ])
        self.pre_process = pre_process
        self.post_process = post_process

    def forward(self, image):
        image = self.pre_process(image.copy())  # C x H x W
        image = image.to(self.device)
        generated_image = self.generator(image.unsqueeze(dim=0)).squeeze(dim=0).cpu()
        return self.post_process(generated_image)
