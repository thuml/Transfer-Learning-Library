from .imagelist import ImageList
from .office31 import Office31
from .officehome import OfficeHome
from .visda2017 import VisDA2017
from .officecaltech import OfficeCaltech
from .domainnet import DomainNet
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .pacs import PACS
from .digits import *
from .aircrafts import Aircraft
from .cub200 import CUB200
from .stanford_cars import StanfordCars
from .stanford_dogs import StanfordDogs
from .coco70 import COCO70
from .oxfordpets import OxfordIIITPets
from .dtd import DTD
from .oxfordflowers import OxfordFlowers102
from .patchcamelyon import PatchCamelyon
from .retinopathy import Retinopathy
from .eurosat import EuroSAT
from .resisc45 import Resisc45
from .food101 import Food101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .cifar import CIFAR10, CIFAR100

__all__ = ['ImageList', 'Office31', 'OfficeHome', "VisDA2017", "OfficeCaltech", "DomainNet", "ImageNetR",
           "ImageNetSketch", "Aircraft", "cub200", "StanfordCars", "StanfordDogs", "COCO70", "OxfordIIITPets", "PACS",
           "DTD", "OxfordFlowers102", "PatchCamelyon", "Retinopathy", "EuroSAT", "Resisc45", "Food101", "SUN397",
           "Caltech101", "CIFAR10", "CIFAR100"]
