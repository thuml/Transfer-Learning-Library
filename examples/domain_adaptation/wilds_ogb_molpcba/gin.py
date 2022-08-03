"""
Adapted from "https://github.com/p-lambda/wilds"
@author: Jiaxin Li
@contact: thulijx@gmail.com
"""
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

__all__ = ['gin_virtual']


class GINVirtual(torch.nn.Module):
    """
    Graph Isomorphism Network augmented with virtual node for multi-task binary graph classification.

    Args:
        num_tasks (int): number of binary label tasks. default to 128 (number of tasks of ogbg-molpcba)
        num_layers (int): number of message passing layers of GNN
        emb_dim (int): dimensionality of hidden channels
        dropout (float): dropout ratio applied to hidden channels

    Inputs:
        - batched Pytorch Geometric graph object

    Outputs:
        - prediction (tensor): float torch tensor of shape (num_graphs, num_tasks)
    """

    def __init__(self, num_tasks=128, num_layers=5, emb_dim=300, dropout=0.5):
        super(GINVirtual, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        if num_tasks is None:
            self.d_out = self.emb_dim
        else:
            self.d_out = self.num_tasks

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        self.gnn_node = GINVirtualNode(num_layers, emb_dim, dropout=dropout)

        # Pooling function to generate whole-graph embeddings
        self.pool = global_mean_pool
        if num_tasks is None:
            self.graph_pred_linear = None
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        if self.graph_pred_linear is None:
            return h_graph
        else:
            return self.graph_pred_linear(h_graph)


class GINVirtualNode(torch.nn.Module):
    """
    Helper function of Graph Isomorphism Network augmented with virtual node for multi-task binary graph classification
    This will generate node embeddings.

    Args:
        num_layers (int): number of message passing layers of GNN
        emb_dim (int): dimensionality of hidden channels
        dropout (float, optional): dropout ratio applied to hidden channels. Default: 0.5

    Inputs:
        - batched Pytorch Geometric graph object
    Outputs:
        - node_embedding (tensor): float torch tensor of shape (num_nodes, emb_dim)
    """

    def __init__(self, num_layers, emb_dim, dropout=0.5):
        super(GINVirtualNode, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        # batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        # List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(2 * emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                    torch.nn.ReLU()))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            # add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            h_list.append(h)

            # update the virtual nodes
            if layer < self.num_layers - 1:
                # add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                # transform virtual nodes using MLP
                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                  self.dropout, training=self.training)

        node_embedding = h_list[-1]

        return node_embedding


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network message passing.

    Args:
        emb_dim (int): node embedding dimensionality

    Inputs:
        - x (tensor): node embedding
        - edge_index (tensor): edge connectivity information
        - edge_attr (tensor): edge feature
    Outputs:
        - prediction (tensor): output node embedding
    """

    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


def gin_virtual(num_tasks, dropout=0.5):
    model = GINVirtual(num_tasks=num_tasks, dropout=dropout)
    return model
