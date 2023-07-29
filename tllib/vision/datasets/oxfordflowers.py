"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OxfordFlowers102(ImageList):
    """
    `The Oxford Flowers 102 <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ is a \
         consistent of 102 flower categories commonly occurring in the United Kingdom. \
         Each class consists of between 40 and 258 images. The images have large scale, \
         pose and light variations. In addition, there are categories that have large \
         variations within the category and several very similar categories. \
         The dataset is divided into a training set, a validation set and a test set. \
         The training set and validation set each consist of 10 images per class \
         (totalling 1020 images each). \
         The test set consists of the remaining 6149 images (minimum 20 per class).

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/161c7b222d6745408201/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/59b6a3fa3dac4404aa3b/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/ec77da479dfb471982fb/?dl=1")
    ]
    CLASSES = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
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

    def __init__(self, root, split='train', download=False, **kwargs):
        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))
        super(OxfordFlowers102, self).__init__(root, OxfordFlowers102.CLASSES,
                                               os.path.join(root, 'image_list', '{}.txt'.format(split)), **kwargs)
