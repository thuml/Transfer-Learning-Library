import torch.nn as nn
from .aspp import DeepLabHead
from .resnet_dilated import resnet_dilated
from tllib.modules.multi_output_module import MultiOutputModule


class HardParameterSharingModel(MultiOutputModule):
    def __init__(self, num_out_channels):
        encoder = resnet_dilated('resnet50')
        decoders = nn.ModuleDict({
            task_name: DeepLabHead(2048, num_out_channels[task_name]) for task_name in num_out_channels
        })
        super().__init__(encoder, decoders)
