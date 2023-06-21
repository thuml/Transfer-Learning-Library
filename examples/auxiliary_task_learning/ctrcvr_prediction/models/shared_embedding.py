import torch
import torch.nn as nn

from tllib.modules.multi_output_module import MultiOutputModule
from .layers import MixedEmbeddingLayer, MultiLayerPerceptron


class SharedEmbeddingModel(MultiOutputModule):
    """
    A class representing a shared embedding model

    Args:
        categorical_field_dims (list): The dimensions of categorical fields.
        numerical_num (int): The number of numerical inputs.
        embed_dim (int): The dimension of the embedding layer.
        tower_mlp_dims (list): The dimensions of the multi-layer perceptron (MLP) layers in the task-specific heads.
        task_names (list): A list of task names.
        dropout (float): The dropout rate.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, tower_mlp_dims, task_names, dropout):
        shared_backbone = MixedEmbeddingLayer(categorical_field_dims, numerical_num, embed_dim)
        task_specific_heads = torch.nn.ModuleDict({
            task_name: nn.Sequential(
                MultiLayerPerceptron(shared_backbone.embed_output_dim, tower_mlp_dims, dropout),
                nn.Sigmoid(),
            )
            for task_name in task_names
        })
        super(SharedEmbeddingModel, self).__init__(shared_backbone, task_specific_heads)
