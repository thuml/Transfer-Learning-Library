import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                                   device: torch.device) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)

    Returns:
        Features in shape (len(data_loader), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.to(device)
            feature = feature_extractor(images).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0)