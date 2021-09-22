import torch
from torch.utils.data import TensorDataset
import os


class TimeSeriesDataset(TensorDataset):
    def __init__(self, data, feature_names, class_names):
        self.feature_names = feature_names
        self.classes = class_names
        super(TimeSeriesDataset, self).__init__(data["x"].transpose(1, 2), data["y"].long())

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @property
    def num_features(self) -> int:
        """Number of features"""
        return len(self.feature_names)
