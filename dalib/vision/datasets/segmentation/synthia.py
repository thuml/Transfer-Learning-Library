import os
from .imagelist import ImageList
from .cityscapes import Cityscapes


class Synthia(ImageList):
    ID_TO_TRAIN_ID = {
        3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
        15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
        8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18
    }
    # classes used to evaluate
    EVALUATE_CLASSES = [
        'road', 'sidewalk', 'building', 'traffic light', 'traffic sign',
        'vegetation', 'sky', 'person', 'rider', 'car', 'bus', 'motorcycle', 'bicycle'
    ]

    # TODO download txt

    def __init__(self, root, split='train', data_folder='RGB', label_folder='synthia_mapped_to_cityscapes', **kwargs):
        data_list_file = os.path.join(root, "image_list", "{}.txt".format(split))
        super(Synthia, self).__init__(root, Cityscapes.CLASSES, data_list_file, data_list_file, data_folder,
                                      label_folder, id_to_train_id=Synthia.ID_TO_TRAIN_ID,
                                      train_id_to_color=Cityscapes.TRAIN_ID_TO_COLOR, **kwargs)
