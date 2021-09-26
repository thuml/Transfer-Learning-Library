"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import sys
import timm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torchvision.transforms as T

sys.path.append('../../..')
from common.utils.metric.reid import extract_reid_feature
from common.utils.analysis import tsne
from common.vision.transforms import RandomErasing
import common.vision.models.reid as models


def copy_state_dict(model, state_dict, strip=None):
    """Copy state dict into the passed in ReID model. As we are using classification loss, which means we need to output
    different number of classes(identities) for different datasets, we will not copy the parameters of last `fc` layer.
    """
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=True)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_train_transform(height, width, resizing='default', random_horizontal_flip=True, random_color_jitter=False,
                        random_gray_scale=False, random_erasing=False):
    """
    resizing mode:
        - default: resize the image to (height, width), zero-pad it by 10 on each size, the take a random crop of
            (height, width)
        - res: resize the image to(height, width)
    """
    if resizing == 'default':
        transform = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width))
        ])
    elif resizing == 'res':
        transform = T.Resize((height, width), interpolation=3)
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
    if random_gray_scale:
        transforms.append(T.RandomGrayscale())
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if random_erasing:
        transforms.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
    return T.Compose(transforms)


def get_val_transform(height, width):
    return T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def visualize_tsne(source_loader, target_loader, model, filename, device, n_data_points_per_domain=3000):
    """Visualize features from different domains using t-SNE. As we can have very large number of samples in each
    domain, only `n_data_points_per_domain` number of samples are randomly selected in each domain.
    """
    source_feature_dict = extract_reid_feature(source_loader, model, device, normalize=True)
    source_feature = torch.stack(list(source_feature_dict.values())).cpu()
    source_feature = source_feature[torch.randperm(len(source_feature))]
    source_feature = source_feature[:n_data_points_per_domain]

    target_feature_dict = extract_reid_feature(target_loader, model, device, normalize=True)
    target_feature = torch.stack(list(target_feature_dict.values())).cpu()
    target_feature = target_feature[torch.randperm(len(target_feature))]
    target_feature = target_feature[:n_data_points_per_domain]

    tsne.visualize(source_feature, target_feature, filename, source_color='cornflowerblue', target_color='darkorange')
    print('T-SNE process is done, figure is saved to {}'.format(filename))


def k_reciprocal_neigh(initial_rank, i, k1):
    """Compute k-reciprocal neighbors of i-th sample. Two samples f_i, f_j are k reciprocal-neighbors if and only if
    each one of them is among the k-nearest samples of another sample.
    """
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = torch.nonzero(backward_k_neigh_index == i)[:, 0]
    return forward_k_neigh_index[fi]


def compute_rerank_dist(target_features, k1=30, k2=6):
    """Compute distance according to `Re-ranking Person Re-identification with k-reciprocal Encoding
    (CVPR 2017) <https://arxiv.org/pdf/1701.08398.pdf>`_.
    """
    n = target_features.size(0)
    original_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True) * 2
    original_dist = original_dist.expand(n, n) - 2 * torch.mm(target_features, target_features.t())
    original_dist /= original_dist.max(0)[0]
    original_dist = original_dist.t()
    initial_rank = torch.argsort(original_dist, dim=-1)
    all_num = gallery_num = original_dist.size(0)

    del target_features

    nn_k1 = []
    nn_k1_half = []
    for i in range(all_num):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = torch.zeros(all_num, all_num)
    for i in range(all_num):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index, candidate_k_reciprocal_index))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)
        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)

    if k2 != 1:
        k2_rank = initial_rank[:, :k2].clone().view(-1)
        V_qe = V[k2_rank]
        V_qe = V_qe.view(initial_rank.size(0), k2, -1).sum(1)
        V_qe /= k2
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(gallery_num):
        invIndex.append(torch.nonzero(V[:, i])[:, 0])

    jaccard_dist = torch.zeros_like(original_dist)
    for i in range(all_num):
        temp_min = torch.zeros(1, gallery_num)
        indNonZero = torch.nonzero(V[i, :])[:, 0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + \
                                        torch.min(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    del invIndex
    del V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    return jaccard_dist
