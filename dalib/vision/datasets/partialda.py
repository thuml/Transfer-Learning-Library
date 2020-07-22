from typing import Union, List, Tuple, Any, Optional, ClassVar
from torchvision.datasets import DatasetFolder
from . import ImageList, Office31, OfficeHome, VisDA2017, ImageNetCaltech, CaltechImageNet, OfficeCaltech
from ._util import read_list_from_file


def partial(dataset_class: ClassVar, kept_classes: Union[List, str]) -> ClassVar:
    if not (issubclass(dataset_class, ImageList) or issubclass(dataset_class, DatasetFolder)):
        raise Exception("Only subclass of ImageList or DatasetFolder can be partial")

    if isinstance(kept_classes, str):
        kept_classes = read_list_from_file(kept_classes)

    class PartialDataset(dataset_class):
        def __init__(self, **kwargs):
            super(PartialDataset, self).__init__(**kwargs)
            partial_classes = []
            for c in kept_classes:
                if isinstance(c, int):
                    partial_classes.append(self.classes[c])
                else:
                    partial_classes.append(c)
            assert all([c in self.classes for c in partial_classes])
            samples = []
            for (path, label) in self.samples:
                class_name = self.classes[label]
                if class_name in partial_classes:
                    samples.append((path, label))
            self.samples = samples
            self.partial_classes = partial_classes
            self.partial_classes_idx = [self.class_to_idx[c] for c in partial_classes]

    return PartialDataset


def default_partial(dataset_class: ClassVar) -> ClassVar:
    if dataset_class == Office31:
        kept_classes = OfficeCaltech.CLASSES
    elif dataset_class == OfficeHome:
        kept_classes = sorted(OfficeHome.CLASSES)[:25]
    elif dataset_class == VisDA2017:
        kept_classes = sorted(VisDA2017.CLASSES)[:6]
    elif dataset_class in [ImageNetCaltech, CaltechImageNet]:
        kept_classes = dataset_class.CLASSES
    else:
        raise NotImplementedError("Unknown partial domain adaptation dataset: {}".format(dataset_class.__name__))
    return partial(dataset_class, kept_classes)