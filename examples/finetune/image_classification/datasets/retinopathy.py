from common.vision.datasets import ImageList


class Retinopathy(ImageList):
    download_list = [
        ("image_list", "image_list.zip", ""),
    ]

    CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    def __init__(self, root, split, **kwargs):

        super(Retinopathy, self).__init__(root, Retinopathy.CLASSES, "{}.txt".format(split), **kwargs)