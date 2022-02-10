from pascal_voc_writer import Writer
import matplotlib.pyplot as plt
import os
import json
from shutil import copy
import time
import tqdm

index_to_class_name = {0: 'unlabeled', 1: 'ego vehicle', 2: 'rectification border', 3: 'out of roi', 4: 'static',
                       5: 'dynamic', 6: 'ground', 7: 'road', 8: 'sidewalk', 9: 'parking', 10: 'rail track',
                       11: 'building', 12: 'wall', 13: 'fence', 14: 'guard rail', 15: 'bridge', 16: 'tunnel',
                       17: 'pole', 18: 'polegroup', 19: 'traffic light', 20: 'traffic sign', 21: 'vegetation',
                       22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider', 26: 'car', 27: 'truck', 28: 'bus',
                       29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle', -1: 'license plate'}


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_xml(img_path, img_shape, data, save_path):
    # path, width, height
    writer = Writer(img_path, img_shape[1], img_shape[0])
    for element in data:
        # name, x1, y1, x2, y2
        writer.addObject(element[0], element[1], element[2], element[3], element[4])
    writer.save(save_path)


if __name__ == '__main__':
    gta_dir = 'datasets/synscapes/synscapes/Synscapes'
    if not os.path.exists(gta_dir):
        print("Please put synscapes datasets in: {}".format(gta_dir))
        exit(0)

    save_root = 'datasets/synscapes_detection'
    make_dir(save_root)
    make_dir(os.path.join(save_root, 'Annotations'))
    make_dir(os.path.join(save_root, 'ImageSets', 'Main'))
    make_dir(os.path.join(save_root, 'JPEGImages'))

    start = time.time()
    # start processing
    meta_dir = os.path.join(gta_dir, 'meta')
    img_dir = os.path.join(gta_dir, 'img', 'rgb')
    list_file = os.listdir(meta_dir)

    # construct txt file
    txt_save_path = os.path.join(save_root, 'ImageSets', 'Main', 'trainval.txt')
    with open(txt_save_path, 'w') as f:
        for filepath in tqdm.tqdm(list_file):
            # get image shape
            file_idx = filepath.split('.')[0]
            img_path = os.path.join(img_dir, file_idx + '.png')
            img_shape = plt.imread(img_path).shape
            # get width height
            width, height = img_shape[1], img_shape[0]

            # construct txt file
            txt_save_path = os.path.join(save_root, 'ImageSets', 'Main', 'trainval.txt')
            f.write(file_idx + '\n')

            with open(os.path.join(meta_dir, filepath), 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
                classes = json_data['instance']['class']
                bboxes = json_data['instance']['bbox2d']

                instances = []
                for idx, bbox in bboxes.items():
                    # get class index
                    class_idx = classes[idx]
                    class_name = index_to_class_name[class_idx]
                    # get bbox
                    xmin = float(bbox['xmin'] * width)
                    ymin = float(bbox['ymin'] * height)
                    xmax = float(bbox['xmax'] * width)
                    ymax = float(bbox['ymax'] * height)
                    instances.append((class_name, xmin, ymin, xmax, ymax))

                # construct xml file
                xml_save_path = os.path.join(save_root, 'Annotations', file_idx + '.xml')
                save_xml(img_path, img_shape, instances, xml_save_path)

                # construct jpeg images
                img_save_path = os.path.join(save_root, 'JPEGImages', file_idx + '.jpg')
                copy(img_path, img_save_path)


    end = time.time() - start
    print('Total Time taken: ', end)
