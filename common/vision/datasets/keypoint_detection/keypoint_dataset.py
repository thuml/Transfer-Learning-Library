"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from abc import ABC
import numpy as np
from torch.utils.data.dataset import Dataset
from webcolors import name_to_rgb
import cv2


class KeypointDataset(Dataset, ABC):
    """A generic dataset class for image keypoint detection

    Args:
        root (str): Root directory of dataset
        num_keypoints (int): Number of keypoints
        samples (list): list of data
        transforms (callable, optional): A function/transform that takes in a dict (which contains PIL image and
            its labels) and returns a transformed version. E.g, :class:`~common.vision.transforms.keypoint_detection.Resize`.
        image_size (tuple): (width, height) of the image. Default: (256, 256)
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2
        keypoints_group (dict): a dict that stores the index of different types of keypoints
        colored_skeleton (dict): a dict that stores the index and color of different skeleton
    """
    def __init__(self, root, num_keypoints, samples, transforms=None, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=2, keypoints_group=None, colored_skeleton=None):
        self.root = root
        self.num_keypoints = num_keypoints
        self.samples = samples
        self.transforms = transforms
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.keypoints_group = keypoints_group
        self.colored_skeleton = colored_skeleton

    def __len__(self):
        return len(self.samples)

    def visualize(self, image, keypoints, filename):
        """Visualize an image with its keypoints, and store the result into a file

        Args:
            image (PIL.Image):
            keypoints (torch.Tensor): keypoints in shape K x 2
            filename (str): the name of file to store
        """
        assert self.colored_skeleton is not None

        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
        for (_, (line, color)) in self.colored_skeleton.items():
            for i in range(len(line) - 1):
                start, end = keypoints[line[i]], keypoints[line[i + 1]]
                cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=name_to_rgb(color),
                         thickness=3)
        for keypoint in keypoints:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, name_to_rgb('black'), 1)
        cv2.imwrite(filename, image)

    def group_accuracy(self, accuracies):
        """ Group the accuracy of K keypoints into different kinds.

        Args:
            accuracies (list): accuracy of the K keypoints

        Returns:
            accuracy of ``N=len(keypoints_group)`` kinds of keypoints

        """
        grouped_accuracies = dict()
        for name, keypoints in self.keypoints_group.items():
            grouped_accuracies[name] = sum([accuracies[idx] for idx in keypoints]) / len(keypoints)
        return grouped_accuracies


class Body16KeypointDataset(KeypointDataset, ABC):
    """
    Dataset with 16 body keypoints.
    """
    # TODO: add image
    head = (9,)
    shoulder = (12, 13)
    elbow = (11, 14)
    wrist = (10, 15)
    hip = (2, 3)
    knee = (1, 4)
    ankle = (0, 5)
    all = (12, 13, 11, 14, 10, 15, 2, 3, 1, 4, 0, 5)
    right_leg = (0, 1, 2, 8)
    left_leg = (5, 4, 3, 8)
    backbone = (8, 9)
    right_arm = (10, 11, 12, 8)
    left_arm = (15, 14, 13, 8)

    def __init__(self, root, samples, **kwargs):
        colored_skeleton = {
            "right_leg": (self.right_leg, 'yellow'),
            "left_leg": (self.left_leg, 'green'),
            "backbone": (self.backbone, 'blue'),
            "right_arm": (self.right_arm, 'purple'),
            "left_arm": (self.left_arm, 'red'),
        }
        keypoints_group = {
            "head": self.head,
            "shoulder": self.shoulder,
            "elbow": self.elbow,
            "wrist": self.wrist,
            "hip": self.hip,
            "knee": self.knee,
            "ankle": self.ankle,
            "all": self.all
        }
        super(Body16KeypointDataset, self).__init__(root, 16, samples, keypoints_group=keypoints_group,
                                                    colored_skeleton=colored_skeleton, **kwargs)


class Hand21KeypointDataset(KeypointDataset, ABC):
    """
    Dataset with 21 hand keypoints.
    """
    # TODO: add image
    MCP = (1, 5, 9, 13, 17)
    PIP = (2, 6, 10, 14, 18)
    DIP = (3, 7, 11, 15, 19)
    fingertip = (4, 8, 12, 16, 20)
    all = tuple(range(21))
    thumb = (0, 1, 2, 3, 4)
    index_finger = (0, 5, 6, 7, 8)
    middle_finger = (0, 9, 10, 11, 12)
    ring_finger = (0, 13, 14, 15, 16)
    little_finger = (0, 17, 18, 19, 20)

    def __init__(self, root, samples, **kwargs):
        colored_skeleton = {
            "thumb": (self.thumb, 'yellow'),
            "index_finger": (self.index_finger, 'green'),
            "middle_finger": (self.middle_finger, 'blue'),
            "ring_finger": (self.ring_finger, 'purple'),
            "little_finger": (self.little_finger, 'red'),
        }
        keypoints_group = {
            "MCP": self.MCP,
            "PIP": self.PIP,
            "DIP": self.DIP,
            "fingertip": self.fingertip,
            "all": self.all
        }
        super(Hand21KeypointDataset, self).__init__(root, 21, samples, keypoints_group=keypoints_group,
                                                    colored_skeleton=colored_skeleton, **kwargs)
