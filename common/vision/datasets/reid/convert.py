"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image


def convert_to_pytorch_dataset(dataset, root=None, transform=None, return_idxes=False):
    class ReidDataset(Dataset):
        def __init__(self, dataset, root, transform):
            super(ReidDataset, self).__init__()
            self.dataset = dataset
            self.root = root
            self.transform = transform
            self.return_idxes = return_idxes

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            fname, pid, cid = self.dataset[index]
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            if not self.return_idxes:
                return img, fname, pid, cid
            else:
                return img, fname, pid, cid, index

    return ReidDataset(dataset, root, transform)
