from PIL import Image
import os
import os.path as osp
import tqdm

import tensorflow_datasets as tfds

import common.vision.datasets as datasets


def _convert_from_tensorflow_datasets(tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg',
                                                  label_names=('label', )):
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
                f.write(filename)
                for label_name in label_names:
                    f.write(" {}".format(ex[label_name]))
                f.write("\n")
                index += 1


class TensorFlowDataset(datasets.ImageList):
    def __init__(self, tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg', label_name='label', **kwargs):
        _convert_from_tensorflow_datasets(tensorflow_dataset, info, root, data_dir, list_file, suffix, [label_name, ])
        classes = info.features[label_name].names
        super(TensorFlowDataset, self).__init__(root, classes, osp.join(root, list_file), **kwargs)


