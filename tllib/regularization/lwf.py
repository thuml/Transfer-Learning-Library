"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import tqdm


def collect_pretrain_labels(data_loader, classifier, device):
    source_predictions = []

    classifier.eval()
    with torch.no_grad():
        for i, (x, label) in enumerate(tqdm.tqdm(data_loader)):
            x = x.to(device)
            y_s = classifier(x)
            source_predictions.append(y_s.detach().cpu())
    return torch.cat(source_predictions, dim=0)


class Classifier(nn.Module):
    """A Classifier used in `Learning Without Forgetting (ECCV 2016)
    <https://arxiv.org/abs/1606.09282>`_..

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data.
        num_classes (int): Number of classes.
        head_source (torch.nn.Module): Classifier head of source model.
        head_target (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True


    Inputs:
        - x (tensor): input data fed to backbone

    Outputs:
        - y_s: predictions of source classifier head
        - y_t: predictions of target classifier head

    Shape:
        - Inputs: (b, *) where b is the batch size and * means any number of additional dimensions
        - y_s: (b, N), where b is the batch size and N is the number of classes
        - y_t: (b, N), where b is the batch size and N is the number of classes

    """
    def __init__(self, backbone: nn.Module, num_classes: int,  head_source,
                 head_target: Optional[nn.Module] = None, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1,  finetune=True, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        self.head_source = head_source
        if head_target is None:
            self.head_target = nn.Linear(self._features_dim, num_classes)
        else:
            self.head_target = head_target
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor):
        """"""
        f = self.backbone(x)
        f = self.pool_layer(f)
        y_s = self.head_source(f)
        y_t = self.head_target(self.bottleneck(f))
        if self.training:
            return y_s, y_t
        else:
            return y_t

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            # {"params": self.head_source.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head_target.parameters(), "lr": 1.0 * base_lr},
        ]
        return params
