import torch
import matplotlib
from typing import Optional

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


def multidomain_visualize(features: torch.Tensor, domain_labels: torch.Tensor,
              filename: str, num_domains: Optional[int]=2, 
              fig_title: Optional[str] = None):
    """
    Visualize features from more than 2 domains using t-SNE.

    Args:
        features (tensor): features from the classifer :math:`(minibatch, F)`
        domain_labels (tensor): labels of the domain
        filename (str): the file name to save t-SNE
        num_domains (int): number of domains to discriminate
        fig_title (str): title for the tSNE diagram

    """
    assert num_domains <= 4 # quick fix that will change when style prediciting
    features = features.numpy()

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    colors = ['r', 'b', 'g', 'y']
    domain_colors = []
    for i in range(num_domains):
        domain_colors.append(colors[i])

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    if fig_title:
        plt.title(fig_title)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domain_labels, cmap=col.ListedColormap(domain_colors), s=2)
    plt.savefig(filename)

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color, target_color]), s=2)
    plt.savefig(filename)

# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, 
#               filename: str, source_domain_labels: Optional[torch.Tensor] = None, 
#             target_domain_labels: Optional[torch.Tensor] = None, num_domains: Optional[int]=2):
#     """
#     Visualize features from different domains using t-SNE.

#     Args:
#         source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
#         target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
#         filename (str): the file name to save t-SNE
#         source_color (str): the color of the source features. Default: 'r'
#         target_color (str): the color of the target features. Default: 'b'

#     """
#     assert num_domains <= 4 # quick fix that will change when color prediciting
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     features = np.concatenate([source_feature, target_feature], axis=0)

#     # map features to 2-d using TSNE
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    
#     if source_domain_labels == None and target_domain_labels == None:
#         # domain labels, 1 represents source while 0 represents target
#         domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
#     else:
#         # domain labels defined by the user since their could be more than 1
#         domains = np.concatenate((source_domain_labels, target_domain_labels))

#     colors = ['r', 'b', 'g', 'y']
#     domain_colors = []
#     for i in range(num_domains):
#         domain_colors.append(colors[i])

#     # visualize using matplotlib
#     plt.figure(figsize=(10, 10))
#     # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color, target_color]), s=2)
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap(domain_colors), s=2)
#     plt.savefig(filename)
