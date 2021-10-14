"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import sys
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.append('../../..')
from common.utils.metric.reid import extract_reid_feature
from common.utils.analysis import tsne
import common.vision.models.reid as models


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
                        random_gray_scale=False):
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
