from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class CaltechImageNet(ImageList):
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/904af0171c82403c9a36/?dl=1"),
        ("256_ObjectCategories", "256_ObjectCategories.tar",
         "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"),
    ]
    image_list = {
        "C": "image_list/caltech_256_list.txt",
        "I": "image_list/imagenet_val_84_list.txt",
    }
    CLASSES = ['ak47', 'american flag', 'backpack', 'baseball bat', 'baseball glove', 'basketball hoop', 'bat',
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

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))
        # TODO check whether root has following sub-directories (train, val, 256_ObjectCategories)

        super(CaltechImageNet, self).__init__(root, CaltechImageNet.CLASSES, data_list_file=data_list_file, **kwargs)


class CaltechImageNetOpenset(ImageList):
    image_list = {
        "C": "image_list/caltech_256_list.txt",
        "I": "image_list/imagenet_val_85_list.txt",
    }
    CLASSES = CaltechImageNet.CLASSES + ['unknown']
    download_list = CaltechImageNet.download_list

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))
        # TODO check whether root has following sub-directories (train, val, 256_ObjectCategories)

        super(CaltechImageNetOpenset, self).__init__(root, CaltechImageNetOpenset.CLASSES, data_list_file=data_list_file, **kwargs)
