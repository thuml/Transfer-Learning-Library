from PIL import Image
import numpy as np
import os
import os.path as osp
import tqdm

import tensorflow_datasets as tfds

import common.vision.datasets as datasets


def _convert_from_tensorflow_datasets(tensorflow_dataset, info, root, data_dir,
                                      list_file, suffix='jpg', labeling_function=None):
    if osp.exists(osp.join(root, list_file)):
        print("Already exists {}. Pass.".format(osp.join(root, list_file)))
    else:
        print("convert from {} to ImageList".format(info.name))
        os.makedirs(osp.join(root, data_dir), exist_ok=True)
        with open(osp.join(root, list_file), "w") as f:
            index = 0
            for ex in tqdm.tqdm(tfds.as_numpy(tensorflow_dataset)):
                image = ex['image']
                if image.shape[2] == 1:
                    image = image.repeat(3, 2)
                im = Image.fromarray(image)
                filename = osp.join(data_dir, "{}.{}".format(index, suffix))
                im.save(osp.join(root, filename))
                f.write("{} {}\n".format(filename, labeling_function(ex)))
                index += 1


class ClevrCount(datasets.ImageList):
    def __init__(self, tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg', **kwargs):
        def _count(sample):
            return len(sample['objects']['size']) - 3
        _convert_from_tensorflow_datasets(tensorflow_dataset, info, root, data_dir, list_file, suffix, _count)
        classes = list(map(str, range(8)))
        super(ClevrCount, self).__init__(root, classes, osp.join(root, list_file), **kwargs)


class ClevrDistance(datasets.ImageList):
    def __init__(self, tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg', **kwargs):
        def _close_distance(sample):
            if len(sample['objects']['pixel_coords']) == 0:
                print(sample['objects'])
                return -100
            dist = np.min(sample['objects']['pixel_coords'][:, 2])
            thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
            label = np.max(np.where((thrs - dist) < 0))
            return label

        _convert_from_tensorflow_datasets(tensorflow_dataset, info, root, data_dir, list_file, suffix, _close_distance)
        classes = list(map(str, [0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0]))
        super(ClevrDistance, self).__init__(root, classes, osp.join(root, list_file), **kwargs)