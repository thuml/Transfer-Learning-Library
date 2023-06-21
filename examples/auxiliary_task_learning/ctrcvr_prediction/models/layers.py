import torch
import torch.nn as nn
import numpy as np


class EmbeddingLayer(torch.nn.Module):
    """
    A class representing an embedding layer.

    Args:
        field_dims (list): The dimensions of the input fields.
        embed_dim (int): The dimension of the embedding layer.
    """
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MixedEmbeddingLayer(nn.Module):
    """
    A class representing a mixed embedding layer that combines categorical and numerical embeddings.

    Args:
        categorical_field_dims (list): The dimensions of categorical fields.
        numerical_num (int): The number of numerical inputs.
        embed_dim (int): The dimension of the embeddings.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim

    def forward(self, x):
        categorical_x, numerical_x = x
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        return torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)


class MultiLayerPerceptron(torch.nn.Module):
    """
    A class representing a multi-layer perceptron (MLP) module.

    Args:
        input_dim (int): The input dimension of the MLP.
        embed_dims (list): The dimensions of the MLP layers.
        dropout (float): The dropout rate.
        output_layer (bool): Whether to include an output layer. Default is True.
    """

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)