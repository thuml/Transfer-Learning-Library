from typing import Union, List, Tuple, Any, Optional, ClassVar, Sequence
from copy import deepcopy
from torchvision.datasets import DatasetFolder
from . import ImageList, Office31, OfficeHome, VisDA2017, OfficeCaltech, ImageNetCaltechOpenset, CaltechImageNetOpenset
from ._util import read_list_from_file


def openset(dataset_class: ClassVar, public_classes: Union[Sequence, str],
            private_classes: Optional[Union[Sequence, str]] = None) -> ClassVar:
    if not (issubclass(dataset_class, ImageList) or issubclass(dataset_class, DatasetFolder)):
        raise Exception("Only subclass of ImageList or DatasetFolder can be openset")

    if isinstance(public_classes, str):
        public_classes = read_list_from_file(public_classes)
    if isinstance(private_classes, str):
        private_classes = read_list_from_file(private_classes)
    elif private_classes is None:
        private_classes = []

    class OpensetDataset(dataset_class):
        def __init__(self, **kwargs):
            super(OpensetDataset, self).__init__(**kwargs)
            for i, c in enumerate(public_classes):
                if isinstance(c, int):
                    public_classes[i] = self.classes[c]
            for i, c in enumerate(private_classes):
                if isinstance(c, int):
                    private_classes[i] = self.classes[c]
            samples = []
            unknown_class_label = len(public_classes)
            for (path, label) in self.samples:
                class_name = self.classes[label]
                if class_name in public_classes:
                    samples.append((path, public_classes.index(class_name)))
                elif class_name in private_classes:
                    samples.append((path, unknown_class_label))
            self.samples = samples
            self.classes = list(deepcopy(public_classes))
            if len(private_classes) > 0:
                self.classes.append('unknown')
            self.class_to_idx = {cls: idx
                                 for idx, cls in enumerate(self.classes)}

    return OpensetDataset


def default_openset(dataset_class: ClassVar, source: bool) -> ClassVar:
    if dataset_class == Office31:
        public_classes = Office31.CLASSES[:20]
        if source:
            private_classes = None
        else:
            private_classes = Office31.CLASSES[20:]
    elif dataset_class == VisDA2017:
        public_classes = ('bicycle', 'bus', 'car', 'motorcycle', 'train', 'truck')
        if source:
            private_classes = None
        else:
            private_classes = ('aeroplane', 'horse', 'knife', 'person', 'plant', 'skateboard')
    else:
        raise NotImplementedError("Unknown openset domain adaptation dataset: {}".format(dataset_class.__name__))
    return openset(dataset_class, public_classes, private_classes)