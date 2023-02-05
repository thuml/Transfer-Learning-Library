"""
@author: Yifei Ji
@contact: jiyf990330@163.com
"""
import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class StanfordDogs(ImageList):
    """`The Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ \
    contains 20,580 images of 120 breeds of dogs from around the world. \
    Each category is composed of exactly 100 training examples and around 72 testing examples.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        sample_rate (int): The sampling rates to sample random ``training`` images for each category.
            Choices include 100, 50, 30, 15. Default: 100.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            train/
            test/
            image_list/
                train_100.txt
                train_50.txt
                train_30.txt
                train_15.txt
                test.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/648df023a6454535887f/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/9791d3d79013479fa189/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/b35375f7349e45db9d11/?dl=1"),
    ]
    image_list = {
        "train": "image_list/train_100.txt",
        "train100": "image_list/train_100.txt",
        "train50": "image_list/train_50.txt",
        "train30": "image_list/train_30.txt",
        "train15": "image_list/train_15.txt",
        "test": "image_list/test.txt",
        "test100": "image_list/test.txt",
    }
    CLASSES = ['n02085620-Chihuahua', 'n02085782-Japanese_spaniel', 'n02085936-Maltese_dog', 'n02086079-Pekinese', 'n02086240-Shih-Tzu',
               'n02086646-Blenheim_spaniel', 'n02086910-papillon', 'n02087046-toy_terrier', 'n02087394-Rhodesian_ridgeback',
               'n02088094-Afghan_hound', 'n02088238-basset', 'n02088364-beagle', 'n02088466-bloodhound', 'n02088632-bluetick', 'n02089078-black-and-tan_coonhound',
               'n02089867-Walker_hound', 'n02089973-English_foxhound', 'n02090379-redbone', 'n02090622-borzoi', 'n02090721-Irish_wolfhound', 'n02091032-Italian_greyhound',
               'n02091134-whippet', 'n02091244-Ibizan_hound', 'n02091467-Norwegian_elkhound', 'n02091635-otterhound', 'n02091831-Saluki', 'n02092002-Scottish_deerhound',
               'n02092339-Weimaraner', 'n02093256-Staffordshire_bullterrier', 'n02093428-American_Staffordshire_terrier', 'n02093647-Bedlington_terrier', 'n02093754-Border_terrier',
               'n02093859-Kerry_blue_terrier', 'n02093991-Irish_terrier', 'n02094114-Norfolk_terrier', 'n02094258-Norwich_terrier', 'n02094433-Yorkshire_terrier',
               'n02095314-wire-haired_fox_terrier', 'n02095570-Lakeland_terrier', 'n02095889-Sealyham_terrier', 'n02096051-Airedale', 'n02096177-cairn', 'n02096294-Australian_terrier',
               'n02096437-Dandie_Dinmont', 'n02096585-Boston_bull', 'n02097047-miniature_schnauzer', 'n02097130-giant_schnauzer', 'n02097209-standard_schnauzer',
               'n02097298-Scotch_terrier', 'n02097474-Tibetan_terrier', 'n02097658-silky_terrier', 'n02098105-soft-coated_wheaten_terrier', 'n02098286-West_Highland_white_terrier',
               'n02098413-Lhasa', 'n02099267-flat-coated_retriever', 'n02099429-curly-coated_retriever', 'n02099601-golden_retriever', 'n02099712-Labrador_retriever',
               'n02099849-Chesapeake_Bay_retriever', 'n02100236-German_short-haired_pointer', 'n02100583-vizsla', 'n02100735-English_setter', 'n02100877-Irish_setter',
               'n02101006-Gordon_setter', 'n02101388-Brittany_spaniel', 'n02101556-clumber', 'n02102040-English_springer', 'n02102177-Welsh_springer_spaniel', 'n02102318-cocker_spaniel',
               'n02102480-Sussex_spaniel', 'n02102973-Irish_water_spaniel', 'n02104029-kuvasz', 'n02104365-schipperke', 'n02105056-groenendael', 'n02105162-malinois', 'n02105251-briard', 'n02105412-kelpie',
               'n02105505-komondor', 'n02105641-Old_English_sheepdog', 'n02105855-Shetland_sheepdog', 'n02106030-collie', 'n02106166-Border_collie', 'n02106382-Bouvier_des_Flandres', 'n02106550-Rottweiler',
               'n02106662-German_shepherd', 'n02107142-Doberman', 'n02107312-miniature_pinscher', 'n02107574-Greater_Swiss_Mountain_dog',
               'n02107683-Bernese_mountain_dog', 'n02107908-Appenzeller', 'n02108000-EntleBucher', 'n02108089-boxer', 'n02108422-bull_mastiff', 'n02108551-Tibetan_mastiff',
               'n02108915-French_bulldog', 'n02109047-Great_Dane', 'n02109525-Saint_Bernard', 'n02109961-Eskimo_dog', 'n02110063-malamute', 'n02110185-Siberian_husky',
               'n02110627-affenpinscher', 'n02110806-basenji', 'n02110958-pug', 'n02111129-Leonberg', 'n02111277-Newfoundland', 'n02111500-Great_Pyrenees', 'n02111889-Samoyed', 'n02112018-Pomeranian',
               'n02112137-chow', 'n02112350-keeshond', 'n02112706-Brabancon_griffon', 'n02113023-Pembroke', 'n02113186-Cardigan',
               'n02113624-toy_poodle', 'n02113712-miniature_poodle', 'n02113799-standard_poodle', 'n02113978-Mexican_hairless', 'n02115641-dingo', 'n02115913-dhole', 'n02116738-African_hunting_dog']


    def __init__(self, root: str, split: str, sample_rate: Optional[int] =100, download: Optional[bool] = False, **kwargs):

        if split == 'train':
            list_name = 'train' + str(sample_rate)
            assert list_name in self.image_list
            data_list_file = os.path.join(root, self.image_list[list_name])
        else:
            data_list_file = os.path.join(root, self.image_list['test'])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(StanfordDogs, self).__init__(root, StanfordDogs.CLASSES, data_list_file=data_list_file, **kwargs)