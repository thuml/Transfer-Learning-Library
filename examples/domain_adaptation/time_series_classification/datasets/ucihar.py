from .time_series import TimeSeriesDataset
import torch
import os
from common.utils.data import concatenate


class UCIHAR(TimeSeriesDataset):
    def __init__(self, root, domain, split="train"):
        feature_names = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]
        class_names = [
            "walking", "walking_upstairs", "walking_downstairs",
            "sitting", "standing", "laying",
        ]
        if isinstance(domain, str):
            data = torch.load(os.path.join(root, "{}_{}".format("ucihar", domain), "{}.pth".format(split)))
        else:
            data = []
            for d in domain:
                data.append(torch.load(os.path.join(root, "{}_{}".format("ucihar", d), "{}.pth".format(split))))
            data = concatenate(data)
        super(UCIHAR, self).__init__(feature_names=feature_names, class_names=class_names, data=data)

