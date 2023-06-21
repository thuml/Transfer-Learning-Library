import torch
import torch.nn as nn
from tllib.modules.multi_output_module import MultiOutputModule
from .layers import MixedEmbeddingLayer, MultiLayerPerceptron


class MMoEBottom(nn.Module):
    """
    A class representing the bottom layer of the MMoE model.

    Args:
        categorical_field_dims (list): The dimensions of categorical fields.
        numerical_num (int): The number of numerical inputs.
        embed_dim (int): The dimension of the embedding layer.
        bottom_mlp_dims (list): The dimensions of the multi-layer perceptron (MLP) layers in the bottom layer.
        expert_num (int): The number of experts in the bottom layer.
        dropout (float): The dropout rate.
    """
    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, expert_num, dropout):
        super(MMoEBottom, self).__init__()
        self.embedding_layer = MixedEmbeddingLayer(categorical_field_dims, numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.expert_num = expert_num
        self.expert = torch.nn.ModuleList(
            [MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in
             range(expert_num)])

    def forward(self, x):
        emb = self.embedding_layer(x)
        return emb, torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim=1)


class MMoETower(nn.Module):
    """
    A class representing a tower of the MMoE model.

    Args:
        embed_output_dim (int): The dimension of the output from the MMoEBottom layer.
        bottom_output_dim (int): The dimension of the output from the bottom layer MLP in the tower.
        tower_mlp_dims (list): The dimensions of the multi-layer perceptron (MLP) layers in the tower.
        expert_num (int): The number of experts in the tower.
        dropout (float): The dropout rate.
    """
    def __init__(self, embed_output_dim, bottom_output_dim, tower_mlp_dims, expert_num, dropout):
        super(MMoETower, self).__init__()
        self.tower = MultiLayerPerceptron(bottom_output_dim, tower_mlp_dims, dropout)
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(embed_output_dim, expert_num),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        emb, fea = x
        gate_value = self.gate(emb).unsqueeze(1)
        task_fea = torch.bmm(gate_value, fea).squeeze(1)
        results = torch.sigmoid(self.tower(task_fea))
        return results


class MMoEModel(MultiOutputModule):
    """
    A pytorch implementation of MMoE Model.

    Reference:
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.

    Args:
        categorical_field_dims (list): The dimensions of categorical fields.
        numerical_num (int): The number of numerical inputs.
        embed_dim (int): The dimension of the embedding layer.
        bottom_mlp_dims (list): The dimensions of the multi-layer perceptron (MLP) layers in the bottom layer.
        tower_mlp_dims (list): The dimensions of the multi-layer perceptron (MLP) layers in the tower.
        task_names (list): A list of task names.
        expert_num (int): The number of experts in the model.
        dropout (float): The dropout rate.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_names,
                 expert_num, dropout):
        bottom = MMoEBottom(categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, expert_num, dropout)
        towers = nn.ModuleDict({
            task_name: MMoETower(bottom.embed_output_dim, bottom_mlp_dims[-1], tower_mlp_dims, expert_num, dropout)
            for task_name in task_names
        })
        super(MMoEModel, self).__init__(bottom, towers)

