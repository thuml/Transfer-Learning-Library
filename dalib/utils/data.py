from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from collections import Counter
from typing import Dict


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def get_frequency_by_category(dataset: Dataset) -> Dict[int, int]:
    r"""
    Get the frequency for each category in a certain dataset.

    Output:
        - **freq** (dict[int, int]): Frequencies of each category appearing in the dataset

    """
    labels = [label for (data, label) in dataset]
    freq = Counter(labels)
    return freq

