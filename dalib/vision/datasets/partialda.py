from typing import Union, List, Tuple, Any, Optional, ClassVar
from torchvision.datasets import DatasetFolder
from . import ImageList, Office31, OfficeHome
from ._util import read_list_from_file


def partial(dataset_class: ClassVar, kept_classes: Optional[Union[List, str]] = None) -> ClassVar:
    if not (issubclass(dataset_class, ImageList) or issubclass(dataset_class, DatasetFolder)):
        raise Exception("Only subclass of ImageList or DatasetFolder can be partial")

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
            samples = []
            for (path, label) in self.samples:
                class_name = self.classes[label]
                if class_name in kept_classes:
                    samples.append((path, label))
            self.samples = samples
            self.partial_classes = kept_classes
            self.partial_classes_idx = [self.class_to_idx[c] for c in kept_classes]

    return PartialClass


DefaultPartialClasses = {
    "Office31": ('back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
                            'laptop_computer', 'monitor', 'mouse', 'mug', 'projector'),
    "OfficeHome": sorted(OfficeHome.CLASSES)[:25],
    "Caltech256": ('ak47', 'backpack', 'bathtub', 'beer mug', 'binoculars', 'camel', 'cannon', 'canoe', 'car tire',
                   'centipede', 'chimp', 'cockroach', 'coffee mug', 'computer keyboard', 'computer monitor', 'conch',
                   'cowboy hat', 'dumb bell', 'electric guitar', 'fighter jet', 'fire truck', 'football helmet',
                   'french horn', 'gas pump', 'golden gate bridge', 'goldfish', 'golf ball', 'goose', 'gorilla',
                   'grand piano', 'grasshopper', 'greyhound', 'hamburger', 'harmonica', 'harp', 'hot air balloon',
                   'hot dog', 'hot tub', 'hourglass', 'house fly', 'hummingbird', 'ice cream cone', 'iguana', 'ipod',
                   'killer whale', 'laptop', 'leopards', 'llama', 'mailbox', 'microwave', 'mountain bike', 'mushroom',
                   'ostrich', 'owl', 'penguin', 'photocopier', 'porcupine', 'praying mantis', 'refrigerator', 'rifle',
                   'school bus', 'scorpion', 'screwdriver', 'self propelled lawn mower', 'skunk', 'snail',
                   'soccer ball', 'socks', 'speed boat', 'spoon', 'starfish', 'syringe', 'teapot', 'tennis ball',
                   'toaster', 'traffic light', 'triceratops', 'tricycle', 'trilobite', 'tripod', 'umbrella',
                   'video projector', 'wine bottle', 'zebra'),
    "ImageNet": (),
}


PartialOffice31 = partial(Office31)

