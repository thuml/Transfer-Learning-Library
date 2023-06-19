"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn
import torch
from typing import Optional, List, Dict


class MultiOutputModule(nn.Module):
    """
    A module that supports multiple outputs by sharing a backbone network.

    Args:
        shared_backbone (nn.Module): The shared backbone network.
        task_specific_heads (nn.ModuleDict): A dictionary of task-specific heads.

    Inputs:
        - x (torch.Tensor): Input tensor.
        - task_name (str): The name of task.

    Outputs:
        - torch.Tensor: Output for the given task.
    """

    def __init__(self, shared_backbone, task_specific_heads):
        super(MultiOutputModule, self).__init__()
        self.shared_backbone = shared_backbone
        self.task_specific_heads = task_specific_heads
        self.grad_numel_list, self.grad_dim = self.compute_grad_dim()

    def forward(self, x: torch.Tensor, task_name=None):
        f = self.shared_backbone(x)
        if task_name is not None:
            return self.task_specific_heads[task_name](f)
        else:
            return {task_name: self.task_specific_heads[task_name](f) for task_name in self.task_specific_heads.keys()}

    def get_shared_parameters(self):
        """
        Get shared parameters of the classifier.

        Returns:
            Generator[torch.Tensor]: Generator object yielding the shared parameters.
        """
        return self.shared_backbone.parameters()

    def compute_grad_dim(self):
        """
        Compute the dimension of the gradients.

        Returns:
            Tuple[List[int], int]: Tuple containing the list of the number of elements in
            the gradients and the total dimension of the gradients.
        """
        grad_numel_list = []
        for param in self.get_shared_parameters():
            grad_numel_list.append(param.data.numel())
        return grad_numel_list, sum(grad_numel_list)

    def get_grad(self):
        """
        Get the 1-dimensional gradient of the shared parameters.

        Returns:
            torch.Tensor: 1-dimensional gradient of the shared parameters.
        """
        grad = torch.zeros(self.grad_dim).to(next(self.parameters()).device)
        cnt = 0
        for param in self.get_shared_parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_numel_list[:cnt])
                end = sum(self.grad_numel_list[:(cnt + 1)])
                grad[beg:end] = param.grad.data.view(-1)
            cnt += 1
        return grad

    def zero_grad_shared_parameters(self):
        """
        Zero the gradient of the shared parameters.
        """
        self.shared_backbone.zero_grad()

    def update_grad(self, grad):
        """
        Update the gradient of the shared parameters.

        Args:
            grad (torch.Tensor): Gradient to be updated.
        """
        cnt = 0
        for param in self.get_shared_parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_numel_list[:cnt])
                end = sum(self.grad_numel_list[:(cnt + 1)])
                param.grad.data = grad[beg:end].contiguous().view(param.data.size()).data.clone()
            cnt += 1


class MultiOutputImageClassifier(MultiOutputModule):
    """
    A multi-output image classifier module.

    Args:
        backbone (torch.nn.Module): Backbone network for feature extraction.
        heads (torch.nn.ModuleDict): Module dictionary containing individual classifier heads for each dataset.
        bottleneck (torch.nn.Module, optional): Bottleneck layer for dimension reduction. Default: None
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        finetune (bool): Whether to finetune the classifier or train from scratch. Default: True
        pool_layer (torch.nn.Module, optional): Pooling layer for feature aggregation. Default: None

    Inputs:
        - x (torch.Tensor): Input tensor.
        - task_name (str): The name of task.

    Outputs:
        - predictions (torch.Tensor): Output for the given task.

    Shape:
        - Inputs: (minibatch, *) where * means any number of additional dimensions.
        - predictions: (minibatch, `num_classes`)

    """
    def __init__(self, backbone: nn.Module, heads: nn.ModuleDict, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1,
                 finetune=True, pool_layer=None):
        if pool_layer is None:
            pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        if bottleneck is None:
            bottleneck = nn.Identity()
            features_dim = backbone.out_features
        else:
            assert bottleneck_dim > 0
            features_dim = bottleneck_dim

        super(MultiOutputImageClassifier, self).__init__(
            shared_backbone=nn.Sequential(
                backbone,
                pool_layer,
                bottleneck
            ),
            task_specific_heads=heads
        )
        self.backbone = backbone
        self.pool_layer = pool_layer
        self.bottleneck = bottleneck
        self._features_dim = features_dim
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.task_specific_heads.parameters(), "lr": 1.0 * base_lr},
        ]

        return params