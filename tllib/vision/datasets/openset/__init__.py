"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from ..imagelist import ImageList
from ..office31 import Office31
from ..officehome import OfficeHome
from ..visda2017 import VisDA2017

from typing import Optional, ClassVar, Sequence
from copy import deepcopy


__all__ = ['Office31', 'OfficeHome', "VisDA2017"]


def open_set(dataset_class: ClassVar, public_classes: Sequence[str],
            private_classes: Optional[Sequence[str]] = ()) -> ClassVar:
    """
    Convert a dataset into its open-set version.

    In other words, those samples which doesn't belong to `private_classes` will be marked as "unknown".

    Be aware that `open_set` will change the label number of each category.

    Args:
        dataset_class (class): Dataset class. Only subclass of ``ImageList`` can be open-set.
        public_classes (sequence[str]): A sequence of which categories need to be kept in the open-set dataset.\
            Each element of `public_classes` must belong to the `classes` list of `dataset_class`.
        private_classes (sequence[str], optional): A sequence of which categories need to be marked as "unknown" \
            in the open-set dataset. Each element of `private_classes` must belong to the `classes` list of \
            `dataset_class`. Default: ().

    Examples::

        >>> public_classes = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard']
        >>> private_classes = ['laptop_computer', 'monitor', 'mouse', 'mug', 'projector']
        >>> # create a open-set dataset class which has classes
        >>> # 'back_pack', 'bike', 'calculator', 'headphones', 'keyboard' and 'unknown'.
        >>> OpenSetOffice31 = open_set(Office31, public_classes, private_classes)
        >>> # create an instance of the open-set dataset
        >>> dataset = OpenSetDataset(root="data/office31", task="A")

    """
    if not (issubclass(dataset_class, ImageList)):
        raise Exception("Only subclass of ImageList can be openset")

    class OpenSetDataset(dataset_class):
        def __init__(self, **kwargs):
            super(OpenSetDataset, self).__init__(**kwargs)
            samples = []
            all_classes = list(deepcopy(public_classes)) + ["unknown"]
            for (path, label) in self.samples:
                class_name = self.classes[label]
                if class_name in public_classes:
                    samples.append((path, all_classes.index(class_name)))
                elif class_name in private_classes:
                    samples.append((path, all_classes.index("unknown")))
            self.samples = samples
            self.classes = all_classes
            self.class_to_idx = {cls: idx
                                 for idx, cls in enumerate(self.classes)}

    return OpenSetDataset


def default_open_set(dataset_class: ClassVar, source: bool) -> ClassVar:
    """
    Default open-set used in some paper.

    Args:
        dataset_class (class): Dataset class. Currently, dataset_class must be one of
            :class:`~tllib.vision.datasets.office31.Office31`, :class:`~tllib.vision.datasets.officehome.OfficeHome`,
            :class:`~tllib.vision.datasets.visda2017.VisDA2017`,
        source (bool): Whether the dataset is used for source domain or not.
    """
    if dataset_class == Office31:
        public_classes = Office31.CLASSES[:20]
        if source:
            private_classes = ()
        else:
            private_classes = Office31.CLASSES[20:]
    elif dataset_class == OfficeHome:
        public_classes = sorted(OfficeHome.CLASSES)[:25]
        if source:
            private_classes = ()
        else:
            private_classes = sorted(OfficeHome.CLASSES)[25:]
    elif dataset_class == VisDA2017:
        public_classes = ('bicycle', 'bus', 'car', 'motorcycle', 'train', 'truck')
        if source:
            private_classes = ()
        else:
            private_classes = ('aeroplane', 'horse', 'knife', 'person', 'plant', 'skateboard')
    else:
        raise NotImplementedError("Unknown openset domain adaptation dataset: {}".format(dataset_class.__name__))
    return open_set(dataset_class, public_classes, private_classes)

