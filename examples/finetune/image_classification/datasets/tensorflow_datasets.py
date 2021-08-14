from PIL import Image
import os
import os.path as osp
import tqdm
import numpy as np

import tensorflow_datasets as tfds

import common.vision.datasets as datasets


def _default_labeling_function(sample):
    return sample['label']


def _convert_from_tensorflow_datasets(dataset_name, root, split, suffix='jpg',
                                      labeling_function=_default_labeling_function):
    list_file = 'imagelist/{}.txt'.format(split)
    if osp.exists(osp.join(root, list_file)):
        print("Already exists {}. Pass.".format(osp.join(root, list_file)))
    else:
        os.makedirs(root, exist_ok=True)
        os.makedirs(osp.join(root, "imagelist"), exist_ok=True)
        # dataset = tfds.builder(dataset_name, data_dir='/data/tensorflow_datasets')
        dataset = tfds.builder(dataset_name)
        data_dir = split
        dataset.download_and_prepare()
        print("convert from {} to ImageList".format(dataset.info.name))
        os.makedirs(osp.join(root, data_dir), exist_ok=True)
        with open(osp.join(root, list_file), "w") as f:
            index = 0
            for ex in tqdm.tqdm(tfds.as_numpy(dataset.as_dataset(split))):
                image = ex['image']
                if image.shape[2] == 1:
                    image = image.repeat(3, 2)
                im = Image.fromarray(image)
                filename = osp.join(data_dir, "{}.{}".format(index, suffix))
                im.save(osp.join(root, filename))
                f.write("{} {}\n".format(filename, labeling_function(ex)))
                index += 1


class Caltech101(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        _convert_from_tensorflow_datasets('caltech101', root, split)
        classes = ['accordion', 'airplanes', 'anchor', 'ant', 'background_google', 'barrel', 'bass', 'beaver',
                   'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon',
                   'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face',
                   'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin',
                   'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'faces', 'faces_easy',
                   'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill',
                   'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch',
                   'lamp', 'laptop', 'leopards', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome',
                   'minaret', 'motorbikes', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza',
                   'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors',
                   'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign',
                   'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair',
                   'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
        super(Caltech101, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class Cifar100(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        _convert_from_tensorflow_datasets('cifar100', root, split)
        classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                   'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                   'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                   'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                   'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
                   'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                   'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
                   'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                   'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                   'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
                   'willow_tree', 'wolf', 'woman', 'worm']
        super(Cifar100, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class DTD(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        _convert_from_tensorflow_datasets('dtd', root, split)
        classes = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked',
                   'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy',
                   'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled',
                   'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous',
                   'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped',
                   'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']
        super(DTD, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class Flowers102(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        _convert_from_tensorflow_datasets('oxford_flowers102', root, split)
        classes = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
                   'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
                   "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower',
                   'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower',
                   'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
                   'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist',
                   'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort',
                   'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
                   'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion',
                   'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura',
                   'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone',
                   'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus',
                   'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple',
                   'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis',
                   'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen', 'watercress',
                   'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia',
                   'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
        super(Flowers102, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class Pets(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        _convert_from_tensorflow_datasets('oxford_iiit_pet', root, split)
        classes = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal',
                   'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau',
                   'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees',
                   'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher',
                   'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard',
                   'samoyed', 'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier',
                   'wheaten_terrier', 'yorkshire_terrier']
        super(Pets, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class PatchCamelyon(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        _convert_from_tensorflow_datasets('patch_camelyon', root, split)
        classes = ['0', '1']
        super(PatchCamelyon, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


# class Sun397(datasets.ImageList):
#     def __init__(self, root, split='train', **kwargs):
#         _convert_from_tensorflow_datasets('sun397', root, split)
#         classes = info.features['label'].names
#         super(Sun397, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class SVHN(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        _convert_from_tensorflow_datasets('svhn_cropped', root, split)
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        super(SVHN, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class DMLab(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        _convert_from_tensorflow_datasets('dmlab', root, split)
        classes = ['0', '1', '2', '3', '4', '5']
        super(DMLab, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class SmallnorbAzimuth(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        def _labeling_function(sample):
            return sample['label_azimuth']
        _convert_from_tensorflow_datasets('smallnorb', root, split, labeling_function=_labeling_function)
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
        super(SmallnorbAzimuth, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class SmallnorblElevation(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        def _labeling_function(sample):
            return sample['label_elevation']
        _convert_from_tensorflow_datasets('smallnorb', root, split, labeling_function=_labeling_function)
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        super(SmallnorblElevation, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class ClevrCount(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        def _count(sample):
            return len(sample['objects']['size']) - 3
        if split == 'test':
            split = 'validation'
        _convert_from_tensorflow_datasets('clevr', root, split, labeling_function=_count)
        classes = ['3', '4', '5', '6', '7', '8', '9', '10']
        super(ClevrCount, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class ClevrDistance(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        def _close_distance(sample):
            if len(sample['objects']['pixel_coords']) == 0:
                print(sample['objects'])
                return -100
            dist = np.min(sample['objects']['pixel_coords'][:, 2])
            thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
            label = np.max(np.where((thrs - dist) < 0))
            return label
        if split == 'test':
            split = 'validation'
        _convert_from_tensorflow_datasets('clevr', root, split, labeling_function=_close_distance)
        classes = ['0.0', '8.0', '8.5', '9.0', '9.5', '10.0', '100.0']
        super(ClevrDistance, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)


class EuroSAT(datasets.ImageList):
    def __init__(self, root, split='train', **kwargs):
        split = 'train[:21600]' if split == 'train' else 'train[21600:]'
        _convert_from_tensorflow_datasets('eurosat', root, split)
        classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture',
                   'PermanentCrop', 'Residential', 'River', 'SeaLake']
        super(EuroSAT, self).__init__(root, classes, osp.join(root, 'imagelist', '{}.txt'.format(split)), **kwargs)

