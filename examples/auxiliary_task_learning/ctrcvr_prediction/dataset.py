import random
import pandas as pd
import numpy as np
import glob
import os

from torch.utils.data import IterableDataset, DataLoader

def wc_count(file_name):
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])


class AliExpressDataset(IterableDataset):
    def __init__(self, dataset_paths, shuffle=False):
        print(dataset_paths)
        self.dataset_paths = dataset_paths
        self.shuffle = shuffle
        self.numerical_num = 63
        self.field_dims = [8, 4, 7, 2, 19, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
        self.size = sum([wc_count(filename) - 1 for filename in self.dataset_paths])

    def __len__(self):
        return self.size

    def _get_iterator(self):
        if self.shuffle:
            random.shuffle(self.dataset_paths)
        for dataset_path in self.dataset_paths:
            data = pd.read_csv(dataset_path).to_numpy()[:, 1:]
            categorical_data = data[:, :16].astype(int)
            numerical_data = data[:, 16: -2].astype(np.float32)
            click_data = data[:, -2].astype(np.float32)
            conversion_data = data[:, -1].astype(np.float32)
            index = list(range(data.shape[0]))
            if self.shuffle:
                random.shuffle(index)
            for i in index:
                yield {
                    "categorical": categorical_data[i],
                    "numerical": numerical_data[i]
                }, {
                    "click": click_data[i],
                    "conversion": conversion_data[i]
                }

    def __iter__(self):
        return self._get_iterator()


def get_datasets(dataset_path, domain_names, batch_size, drop_last=False):
    train_filenames = []
    val_filenames = []
    test_filenames = []

    for domain_name in domain_names:
        train_filenames += sorted(glob.glob(os.path.join(dataset_path, domain_name, "train_*.csv")))
        val_test_filenames = sorted(glob.glob(os.path.join(dataset_path, domain_name, "test_*.csv")))
        val_filenames += val_test_filenames[:int(len(val_test_filenames) * 0.5)]
        test_filenames += val_test_filenames[int(len(val_test_filenames) * 0.5):]
    train_dataset = AliExpressDataset(train_filenames, shuffle=True)
    val_dataset = AliExpressDataset(val_filenames)
    test_dataset = AliExpressDataset(test_filenames)

    # must use 1 workers !
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=drop_last)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    return train_data_loader, val_data_loader, test_data_loader, field_dims, numerical_num

