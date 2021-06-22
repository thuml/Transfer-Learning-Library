import itertools
from torch.utils.data import DataLoader, Dataset
from typing import TypeVar, Iterable


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


class CombineDataset(Dataset[T_co]):
    r"""Dataset as a combination of multiple datasets.

    The element of each dataset must be a list, and the i-th element of the combined dataset
     is a list splicing of the i-th element of each sub dataset.
    The length of the combined dataset is the minimum of the lengths of all sub datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(CombineDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)

    def __len__(self):
        return min([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        return list(itertools.chain(*[d[idx] for d in self.datasets]))

