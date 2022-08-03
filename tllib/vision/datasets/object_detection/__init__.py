"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import numpy as np
import os
import xml.etree.ElementTree as ET

from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
)
from detectron2.utils.file_io import PathManager
from detectron2.structures import BoxMode
from tllib.vision.datasets._util import download as download_dataset


def parse_root_and_file_name(path):
    path_list = path.split('/')
    dataset_root = '/'.join(path_list[:-1])
    file_name = path_list[-1]
    if dataset_root == '':
        dataset_root = '.'
    return dataset_root, file_name


class VOCBase:
    class_names = (
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    )

    def __init__(self, root, split="trainval", year=2007, ext='.jpg', download=True):
        self.name = "{}_{}".format(root, split)
        self.name = self.name.replace(os.path.sep, "_")
        if self.name not in MetadataCatalog.keys():
            register_pascal_voc(self.name, root, split, year, class_names=self.class_names, ext=ext)
            MetadataCatalog.get(self.name).evaluator_type = "pascal_voc"
        if download:
            dataset_root, file_name = parse_root_and_file_name(root)
            download_dataset(dataset_root, file_name, self.archive_name, self.dataset_url)


class VOC2007(VOCBase):
    archive_name = 'VOC2007.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/cc2d40bc5f00445eb05e/?dl=1'

    def __init__(self, root):
        super(VOC2007, self).__init__(root)


class VOC2012(VOCBase):
    archive_name = 'VOC2012.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/29450c5e151843999872/?dl=1'

    def __init__(self, root):
        super(VOC2012, self).__init__(root, year=2012)


class VOC2007Test(VOCBase):
    archive_name = 'VOC2007.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/cc2d40bc5f00445eb05e/?dl=1'

    def __init__(self, root):
        super(VOC2007Test, self).__init__(root, year=2007, split='test')


class Clipart(VOCBase):
    archive_name = 'clipart.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/53ae84b87016418d931d/?dl=1'


class VOCPartialBase:
    class_names = (
        "bicycle", "bird", "car", "cat", "dog", "person",
    )

    def __init__(self, root, split="trainval", year=2007, ext='.jpg', download=True):
        self.name = "{}_{}".format(root, split)
        self.name = self.name.replace(os.path.sep, "_")
        if self.name not in MetadataCatalog.keys():
            register_pascal_voc(self.name, root, split, year, class_names=self.class_names, ext=ext)
            MetadataCatalog.get(self.name).evaluator_type = "pascal_voc"
        if download:
            dataset_root, file_name = parse_root_and_file_name(root)
            download_dataset(dataset_root, file_name, self.archive_name, self.dataset_url)


class VOC2007Partial(VOCPartialBase):
    archive_name = 'VOC2007.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/cc2d40bc5f00445eb05e/?dl=1'

    def __init__(self, root):
        super(VOC2007Partial, self).__init__(root)


class VOC2012Partial(VOCPartialBase):
    archive_name = 'VOC2012.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/29450c5e151843999872/?dl=1'

    def __init__(self, root):
        super(VOC2012Partial, self).__init__(root, year=2012)


class VOC2007PartialTest(VOCPartialBase):
    archive_name = 'VOC2007.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/cc2d40bc5f00445eb05e/?dl=1'

    def __init__(self, root):
        super(VOC2007PartialTest, self).__init__(root, year=2007, split='test')


class WaterColor(VOCPartialBase):
    archive_name = 'watercolor.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/f6b648436ac2497bb232/?dl=1'

    def __init__(self, root):
        super(WaterColor, self).__init__(root, split='train')


class WaterColorTest(VOCPartialBase):
    archive_name = 'watercolor.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/f6b648436ac2497bb232/?dl=1'

    def __init__(self, root):
        super(WaterColorTest, self).__init__(root, split='test')


class Comic(VOCPartialBase):
    archive_name = 'comic.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/a7c38db53287449f9db2/?dl=1'

    def __init__(self, root):
        super(Comic, self).__init__(root, split='train')


class ComicTest(VOCPartialBase):
    archive_name = 'comic.tar'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/a7c38db53287449f9db2/?dl=1'

    def __init__(self, root):
        super(ComicTest, self).__init__(root, split='test')


class CityscapesBase:
    class_names = (
        "bicycle", "bus", "car", "motorcycle", "person", "rider", "train", "truck",
    )

    def __init__(self, root, split="trainval", year=2007, ext='.png'):
        self.name = "{}_{}".format(root, split)
        self.name = self.name.replace(os.path.sep, "_")
        if self.name not in MetadataCatalog.keys():
            register_pascal_voc(self.name, root, split, year, class_names=self.class_names, ext=ext,
                                bbox_zero_based=True)
            MetadataCatalog.get(self.name).evaluator_type = "pascal_voc"


class Cityscapes(CityscapesBase):
    def __init__(self, root):
        super(Cityscapes, self).__init__(root, split="trainval")


class CityscapesTest(CityscapesBase):
    def __init__(self, root):
        super(CityscapesTest, self).__init__(root, split='test')


class FoggyCityscapes(Cityscapes):
    pass


class FoggyCityscapesTest(CityscapesTest):
    pass


class CityscapesCarBase:
    class_names = (
        "car",
    )

    def __init__(self, root, split="trainval", year=2007, ext='.png', bbox_zero_based=True):
        self.name = "{}_{}".format(root, split)
        self.name = self.name.replace(os.path.sep, "_")
        if self.name not in MetadataCatalog.keys():
            register_pascal_voc(self.name, root, split, year, class_names=self.class_names, ext=ext,
                                bbox_zero_based=bbox_zero_based)
            MetadataCatalog.get(self.name).evaluator_type = "pascal_voc"


class CityscapesCar(CityscapesCarBase):
    pass


class CityscapesCarTest(CityscapesCarBase):
    def __init__(self, root):
        super(CityscapesCarTest, self).__init__(root, split='test')


class Sim10kCar(CityscapesCarBase):
    def __init__(self, root):
        super(Sim10kCar, self).__init__(root, split='trainval10k', ext='.jpg', bbox_zero_based=False)


class KITTICar(CityscapesCarBase):
    def __init__(self, root):
        super(KITTICar, self).__init__(root, split='trainval', ext='.jpg', bbox_zero_based=False)


class GTA5(CityscapesBase):
    def __init__(self, root):
        super(GTA5, self).__init__(root, split="trainval", ext='.jpg')


def load_voc_instances(dirname: str, split: str, class_names, ext='.jpg', bbox_zero_based=False):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    skip_classes = set()
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ext)

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in class_names:
                skip_classes.add(cls)
                continue
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            if bbox_zero_based is False:
                bbox[0] -= 1.0
                bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    print("Skip classes:", list(skip_classes))
    return dicts


def register_pascal_voc(name, dirname, split, year, class_names, **kwargs):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names, **kwargs))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
