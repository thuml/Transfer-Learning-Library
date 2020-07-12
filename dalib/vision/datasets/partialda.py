from typing import Union, List, Tuple, Any, Optional, ClassVar

from . import ImageList, Office31, OfficeHome
from ._util import read_list_from_file


def partial(dataset_class: ClassVar, kept_classes: Optional[Union[List, str]] = None) -> ClassVar:
    if not issubclass(dataset_class, ImageList):
        raise Exception("Only subclass of ImageList can be partial")

    if kept_classes is None:
        dataset_name = dataset_class.__name__
        if dataset_name not in DefaultPartialClasses.keys():
            raise NotImplementedError("Unknown partial domain adaptation dataset: {}".format(dataset_name))
        else:
            kept_classes = DefaultPartialClasses[dataset_name]
    elif isinstance(kept_classes, str):
        kept_classes = read_list_from_file(kept_classes)

    class PartialClass(dataset_class):
        def __init__(self, **kwargs):
            super(PartialClass, self).__init__(**kwargs)
            assert all([c in self.classes for c in kept_classes])
            data = []
            for (path, label) in self.data:
                class_name = self.classes[label]
                if class_name in kept_classes:
                    data.append((path, label))
            self.data = data
            self.partial_classes = kept_classes
            self.partial_classes_idx = [self.class_to_idx[c] for c in kept_classes]

    return PartialClass


# def partial(dataset: ImageList, kept_classes: Optional[Union[List, str]] = None) -> Tuple[ImageList, List[str]]:
#     if kept_classes is None:
#         dataset_name = dataset.__class__.__name__
#         if dataset_name not in DefaultPartialClasses.keys():
#             raise NotImplementedError("Unknown partial domain adaptation dataset: {}".format(dataset_name))
#         else:
#             kept_classes = DefaultPartialClasses[dataset_name]
#     elif isinstance(kept_classes, str):
#         kept_classes = read_list_from_file(kept_classes)
#         assert all([c in dataset.classes for c in kept_classes])
#     data = []
#     for (path, label) in dataset.data:
#         class_name = dataset.classes[label]
#         if class_name in kept_classes:
#             data.append((path, label))
#     dataset.data = data
#     return dataset, kept_classes


DefaultPartialClasses = {
    "Office31": ('back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
                            'laptop_computer', 'monitor', 'mouse', 'mug', 'projector'),
    "OfficeHome": sorted(OfficeHome.CLASSES)[:25],

}


PartialOffice31 = partial(Office31)

