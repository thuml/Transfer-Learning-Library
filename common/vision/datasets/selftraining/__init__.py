from ..imagelist import ImageList
from ..office31 import Office31
from ..officehome import OfficeHome
from ..domainnet import DomainNet
from ..visda2017 import VisDA2017
from ..officecaltech import OfficeCaltech

from typing import ClassVar, Tuple, Any

__all__ = ['Office31', 'OfficeHome', 'VisDA2017', 'DomainNet', 'OfficeCaltech']


def perform_multiple_transforms(dataset_class: ClassVar):
    if not (issubclass(dataset_class, ImageList)):
        raise Exception("Only subclass of ImageList can be used as double_input_dataset")

    class MultipleTransformsDataset(dataset_class):
        def __init__(self, **kwargs):
            super(MultipleTransformsDataset, self).__init__(**kwargs)

        def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
            """
            Args:
                index (int): Index
                return (tuple): (image1, image2, target) where image1, image2 both come from same image but
                through different augmentations, target is index of the target class.
            """
            path, target = self.samples[index]
            img = self.loader(path)
            assert self.transform is not None
            img1 = self.transform(img)
            img2 = self.transform(img)
            if self.target_transform is not None and target is not None:
                target = self.target_transform(target)

            return img1, img2, target

    return MultipleTransformsDataset
