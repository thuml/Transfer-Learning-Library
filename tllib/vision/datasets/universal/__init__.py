"""
@author: Jinghan Gao
@contact: getterk@163.com
"""
from ..imagelist import ImageList
from ..office31 import Office31
from ..officehome import OfficeHome
from ..visda2017 import VisDA2017
from ..domainnet import DomainNet

from typing import Optional, ClassVar, Sequence, List
from copy import deepcopy

__all__ = ['Office31', 'OfficeHome', 'VisDA2017', 'DomainNet', 'default_universal']


def universal(dataset_class: ClassVar, public_classes: Sequence[str],
              private_classes: Optional[Sequence[str]] = (), class_mapping: Optional[List] = None) -> ClassVar:
    """
    Convert a dataset into its universal version.

    In other words, those samples which doesn't belong to `private_classes` will be marked as "unknown".

    Args:
        dataset_class (class): Dataset class. Only subclass of ``ImageList`` can be a universal dataset.
        public_classes (sequence[str]): A sequence of which categories need to be kept in the universal dataset.\
            Each element of `public_classes` must belong to the `classes` list of `dataset_class`.
        private_classes (sequence[str], optional): A sequence of which categories that are privately kept \
            in the universal dataset. Each element of `private_classes` must belong to the `classes` list of \
            `dataset_class`. Default: ().
        class_mapping: (list, optional): a class mapping
    """
    if not (issubclass(dataset_class, ImageList)):
        raise Exception("Only subclass of ImageList can be a universal dataset")

    class UniversalDataset(dataset_class):
        def __init__(self, **kwargs):
            super(UniversalDataset, self).__init__(**kwargs)
            samples = []
            all_classes = list(deepcopy(public_classes)) + list(deepcopy(private_classes))
            for (path, label) in self.samples:
                class_name = self.classes[label]
                if class_name in all_classes:
                    if class_mapping:
                        label = class_mapping[label]
                    samples.append((path, label))
            self.samples = samples
            self.targets = [s[1] for s in self.samples]
            self.classes = all_classes
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    return UniversalDataset


def default_universal(dataset_class: ClassVar, source: bool) -> ClassVar:
    """
    Default universal dataset used in some paper.

    Args:
        dataset_class (class): Dataset class. Currently, dataset_class must be one of
            :class:`~tllib.vision.datasets.office31.Office31`, :class:`~tllib.vision.datasets.officehome.OfficeHome`,
            :class:`~tllib.vision.datasets.visda2017.VisDA2017`, :class:`~tllib.vision.datasets.domainnet.DomainNet`,
        source (bool): Whether the dataset is used for source domain or not.
    """
    class_mapping = None
    if dataset_class == Office31:
        public_classes = Office31.CLASSES[:10]
        if source:
            private_classes = Office31.CLASSES[10:20]
        else:
            private_classes = Office31.CLASSES[20:]
    elif dataset_class == OfficeHome:
        sorted_classes = sorted(OfficeHome.CLASSES)
        class_mapping = [sorted_classes.index(c) for c in OfficeHome.CLASSES]
        public_classes = sorted_classes[:10]
        if source:
            private_classes = sorted_classes[10:15]
        else:
            private_classes = sorted_classes[15:]
    elif dataset_class == VisDA2017:
        ordered_classes = ['aeroplane', 'bus', 'horse', 'knife', 'person', 'skateboard', 'truck', 'bicycle', 'car',
                           'motorcycle', 'plant', 'train']
        class_mapping = [ordered_classes.index(c) for c in VisDA2017.CLASSES]
        public_classes = ('aeroplane', 'bus', 'horse', 'knife', 'person', 'skateboard')
        if source:
            private_classes = ('truck', 'bicycle', 'car')
        else:
            private_classes = ('motorcycle', 'plant', 'train')
    elif dataset_class == DomainNet:
        public_classes = DomainNet.CLASSES[:150]
        if source:
            private_classes = DomainNet.CLASSES[150:200]
        else:
            private_classes = DomainNet.CLASSES[200:]
    else:
        raise NotImplementedError("Unknown universal domain adaptation dataset: {}".format(dataset_class.__name__))
    return universal(dataset_class, public_classes, private_classes, class_mapping)
