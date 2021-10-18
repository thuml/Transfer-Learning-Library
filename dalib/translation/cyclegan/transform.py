"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import torch.nn as nn
import torchvision.transforms as T

from common.vision.transforms import Denormalize


class Translation(nn.Module):
    """
    Image Translation Transform Module

    Args:
        generator (torch.nn.Module): An image generator, e.g. :meth:`~dalib.translation.cyclegan.resnet_9_generator`
        device (torch.device): device to put the generator. Default: 'cpu'
        mean (tuple): the normalized mean for image
        std (tuple): the normalized std for image
    Input:
        - image (PIL.Image): raw image in shape H x W x C

    Output:
        raw image in shape H x W x 3

    """
    def __init__(self, generator, device=torch.device("cpu"), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super(Translation, self).__init__()
        self.generator = generator.to(device)
        self.device = device
        self.pre_process = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        self.post_process = T.Compose([
            Denormalize(mean, std),
            T.ToPILImage()
        ])

    def forward(self, image):
        image = self.pre_process(image.copy())  # C x H x W
        image = image.to(self.device)
        generated_image = self.generator(image.unsqueeze(dim=0)).squeeze(dim=0).cpu()
        return self.post_process(generated_image)
