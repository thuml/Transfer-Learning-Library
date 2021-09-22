from .time_series import TimeSeriesDataset
import torch
import os
from common.utils.data import concatenate


class WISDMAR(TimeSeriesDataset):
    def __init__(self, root, domain, split="train"):
        feature_names = [
            "acc_x", "acc_y", "acc_z",
        ]
        class_names = [
            "Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs",
        ]
        if isinstance(domain, str):
            data = torch.load(os.path.join(root, "{}_{}".format("wisdm_ar", domain), "{}.pth".format(split)))
        else:
            data = []
            for d in domain:
                data.append(torch.load(os.path.join(root, "{}_{}".format("wisdm_ar", d), "{}.pth".format(split))))
            data = concatenate(data)
        super(WISDMAR, self).__init__(feature_names=feature_names, class_names=class_names, data=data)

