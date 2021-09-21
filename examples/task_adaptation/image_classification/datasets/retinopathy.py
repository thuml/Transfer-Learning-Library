import os
from common.vision.datasets import ImageList
from common.vision.datasets._util import download


class Retinopathy(ImageList):
    CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    def __init__(self, root, split, **kwargs):
        download(root, "image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/240edc4d3fd248549b8e/?dl=1")
        super(Retinopathy, self).__init__(os.path.join(root, split), Retinopathy.CLASSES, os.path.join(root, "image_list", "{}.txt".format(split)), **kwargs)
