"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import os
from ..imagelist import ImageList
from .._util import download as download_data, check_exits

_CLASSES = ['ak47', 'american flag', 'backpack', 'baseball bat', 'baseball glove', 'basketball hoop', 'bat',
           'bathtub', 'bear', 'beer mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai 101',
           'boom box', 'bowling ball', 'bowling pin', 'boxing glove', 'brain 101', 'breadmaker', 'buddha 101',
           'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car tire',
           'cartman', 'cd', 'centipede', 'cereal box', 'chandelier 101', 'chess board', 'chimp', 'chopsticks',
           'cockroach', 'coffee mug', 'coffin', 'coin', 'comet', 'computer keyboard', 'computer monitor',
           'computer mouse', 'conch', 'cormorant', 'covered wagon', 'cowboy hat', 'crab 101', 'desk globe',
           'diamond ring', 'dice', 'dog', 'dolphin 101', 'doorknob', 'drinking straw', 'duck', 'dumb bell',
           'eiffel tower', 'electric guitar 101', 'elephant 101', 'elk', 'ewer 101', 'eyeglasses', 'fern',
           'fighter jet', 'fire extinguisher', 'fire hydrant', 'fire truck', 'fireworks', 'flashlight',
           'floppy disk', 'football helmet', 'french horn', 'fried egg', 'frisbee', 'frog', 'frying pan',
           'galaxy', 'gas pump', 'giraffe', 'goat', 'golden gate bridge', 'goldfish', 'golf ball', 'goose',
           'gorilla', 'grand piano 101', 'grapes', 'grasshopper', 'guitar pick', 'hamburger', 'hammock',
           'harmonica', 'harp', 'harpsichord', 'hawksbill 101', 'head phones', 'helicopter 101', 'hibiscus',
           'homer simpson', 'horse', 'horseshoe crab', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass',
           'house fly', 'human skeleton', 'hummingbird', 'ibis 101', 'ice cream cone', 'iguana', 'ipod',
           'iris', 'jesus christ', 'joy stick', 'kangaroo 101', 'kayak', 'ketch 101', 'killer whale', 'knife',
           'ladder', 'laptop 101', 'lathe', 'leopards 101', 'license plate', 'lightbulb', 'light house',
           'lightning', 'llama 101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah 101',
           'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes 101', 'mountain bike', 'mushroom',
           'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm pilot', 'palm tree', 'paperclip',
           'paper shredder', 'pci card', 'penguin', 'people', 'pez dispenser', 'photocopier', 'picnic table',
           'playing card', 'porcupine', 'pram', 'praying mantis', 'pyramid', 'raccoon', 'radio telescope',
           'rainbow', 'refrigerator', 'revolver 101', 'rifle', 'rotary phone', 'roulette wheel', 'saddle',
           'saturn', 'school bus', 'scorpion 101', 'screwdriver', 'segway', 'self propelled lawn mower',
           'sextant', 'sheet music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake',
           'sneaker', 'snowmobile', 'soccer ball', 'socks', 'soda can', 'spaghetti', 'speed boat', 'spider',
           'spoon', 'stained glass', 'starfish 101', 'steering wheel', 'stirrups', 'sunflower 101', 'superman',
           'sushi', 'swan', 'swiss army knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy bear',
           'teepee', 'telephone box', 'tennis ball', 'tennis court', 'tennis racket', 'theodolite', 'toaster',
           'tomato', 'tombstone', 'top hat', 'touring bike', 'tower pisa', 'traffic light', 'treadmill',
           'triceratops', 'tricycle', 'trilobite 101', 'tripod', 't shirt', 'tuning fork', 'tweezer',
           'umbrella 101', 'unicorn', 'vcr', 'video projector', 'washing machine', 'watch 101', 'waterfall',
           'watermelon', 'welding mask', 'wheelbarrow', 'windmill', 'wine bottle', 'xylophone', 'yarmulke',
           'yo yo', 'zebra', 'airplanes 101', 'car side 101', 'faces easy 101', 'greyhound', 'tennis shoes',
           'toad']


class CaltechImageNet(ImageList):
    """Caltech-ImageNet is constructed from `Caltech-256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ and
    `ImageNet-1K <http://image-net.org/>`_ .

    They share 84 common classes. Caltech-ImageNet keeps all classes of Caltech-256.
    The label is based on the Caltech256 (class 0-255) . The private classes of ImageNet-1K is discarded.


    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'C'``:Caltech-256, \
            ``'I'``: ImageNet-1K validation set.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: You need to put ``train`` and ``val`` directory of ImageNet-1K manually in `root` directory
        since ImageNet-1K is no longer publicly accessible. DALIB will only download Caltech-256 and ImageList automatically.
        In `root`, there will exist following files after downloading.
        ::
            train/
                n01440764/
                ...
            val/
            256_ObjectCategories/
                001.ak47/
                ...
            image_list/
                caltech_256_list.txt
                ...
    """
    image_list = {
        "C": "image_list/caltech_256_list.txt",
        "I": "image_list/imagenet_val_84_list.txt",
    }
    CLASSES = _CLASSES

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), download_list))

        if not os.path.exists(os.path.join(root, 'val')):
            print("Please put train and val directory of ImageNet-1K manually under {} "
                  "since ImageNet-1K is no longer publicly accessible.".format(root))
            exit(-1)

        super(CaltechImageNet, self).__init__(root, CaltechImageNet.CLASSES, data_list_file=data_list_file, **kwargs)


class CaltechImageNetUniversal(ImageList):
    """Caltech-ImageNet-Universal is constructed from `Caltech-256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_
        and `ImageNet-1K <http://image-net.org/>`_ .

        They share 84 common classes. Caltech-ImageNet keeps all classes of Caltech-256.
        The label is based on the Caltech256 (class 0-255) . The private classes of ImageNet-1K is grouped into class 256 ("unknown").
        Thus, CaltechImageNetUniversal has 257 classes in total.

        Args:
            root (str): Root directory of dataset
            task (str): The task (domain) to create dataset. Choices include ``'C'``:Caltech-256, \
                ``'I'``: ImageNet-1K validation set.
            download (bool, optional): If true, downloads the dataset from the internet and puts it \
                in root directory. If dataset is already downloaded, it is not downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
                transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.

        .. note:: You need to put ``train`` and ``val`` directory of ImageNet-1K manually in `root` directory
            since ImageNet-1K is no longer publicly accessible. DALIB will only download Caltech-256 and ImageList automatically.
            In `root`, there will exist following files after downloading.
            ::
                train/
                    n01440764/
                    ...
                val/
                256_ObjectCategories/
                    001.ak47/
                    ...
                image_list/
                    caltech_256_list.txt
                    ...
        """
    image_list = {
        "C": "image_list/caltech_256_list.txt",
        "I": "image_list/imagenet_val_85_list.txt",
    }
    CLASSES = _CLASSES + ['unknown']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), download_list))

        if not os.path.exists(os.path.join(root, 'val')):
            print("Please put train and val directory of ImageNet-1K manually under {} "
                  "since ImageNet-1K is no longer publicly accessible.".format(root))
            exit(-1)

        super(CaltechImageNetUniversal, self).__init__(root, CaltechImageNetUniversal.CLASSES,
                                                     data_list_file=data_list_file, **kwargs)



download_list = [
    ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/a0d7ea37026946f98965/?dl=1"),
    ("256_ObjectCategories", "256_ObjectCategories.tar",
     "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"),
]


