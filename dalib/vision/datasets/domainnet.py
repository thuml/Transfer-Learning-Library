import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class DomainNet(ImageList):
    """`DomainNet <http://ai.bu.edu/M3SDA/#dataset>`_ (cleaned version, recommended)

    See `Moment Matching for Multi-Source Domain Adaptation <https://arxiv.org/abs/1812.01754>`_ for details.

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'c'``:clipart, \
            ``'i'``: infograph, ``'p'``: painting, ``'q'``: quickdraw, ``'r'``: real, ``'s'``: sketch
        - **evaluate** (bool, optional): If true, use the test set. Otherwise, use the training set. Default: False
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            clipart/
            infograph/
            painting/
            quickdraw/
            real/
            sketch/
            image_list/
                clipart.txt
                ...
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/73489ae10aea45d58194/?dl=1"),
        ("clipart", "clipart.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip"),
        ("infograph", "infograph.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip"),
        ("painting", "painting.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip"),
        ("quickdraw", "quickdraw.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"),
        ("real", "real.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"),
        ("sketch", "sketch.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"),
    ]
    image_list = {
        "c": "clipart",
        "i": "infograph",
        "p": "painting",
        "q": "quickdraw",
        "r": "real",
        "s": "sketch",
    }

    def __init__(self, root, task, evaluate=False, download=False, **kwargs):
        assert task in self.image_list
        phase = "test" if evaluate else "train"
        data_list_file = os.path.join(root, "image_list", "{}_{}.txt".format(self.image_list[task], phase))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda args: check_exits(root, args[0]), self.download_list))

        super(DomainNet, self).__init__(root, num_classes=345, data_list_file=data_list_file, **kwargs)

