"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import copy
import numpy as np
import os
import json
from typing import Optional, Callable, List
import random
import pprint

import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import crop
from detectron2.structures import pairwise_iou
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T


class ProposalMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        origin_image_shape = image.shape[:2]  # h, w

        aug_input = T.AugInput(image)
        image = aug_input.image

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                obj
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, origin_image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


class ProposalGenerator(DatasetEvaluator):
    """
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a ProposalGenerator to generate proposals for each inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and generate proposals results in the end (by :meth:`evaluate`).
    """
    def __init__(self, iou_threshold=(0.4, 0.5), num_classes=20, *args, **kwargs):
        super(ProposalGenerator, self).__init__(*args, **kwargs)
        self.fg_proposal_list = []
        self.bg_proposal_list = []
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes

    def process_type(self, inputs, outputs, type='instances'):
        cpu_device = torch.device('cpu')
        input_instance = inputs[0]['instances'].to(cpu_device)
        output_instance = outputs[0][type].to(cpu_device)
        filename = inputs[0]['file_name']
        pred_boxes = output_instance.pred_boxes
        pred_scores = output_instance.scores
        pred_classes = output_instance.pred_classes
        proposal = Proposal(
            image_id=inputs[0]['image_id'],
            filename=filename,
            pred_boxes=pred_boxes.tensor.numpy(),
            pred_classes=pred_classes.numpy(),
            pred_scores=pred_scores.numpy(),
        )

        if hasattr(input_instance, 'gt_boxes'):
            gt_boxes = input_instance.gt_boxes
            # assign a gt label for each pred_box
            if pred_boxes.tensor.shape[0] == 0:
                proposal.gt_fg_classes = proposal.gt_classes = proposal.gt_ious = proposal.gt_boxes = np.array([])
            elif gt_boxes.tensor.shape[0] == 0:
                proposal.gt_fg_classes = proposal.gt_classes = np.array([self.num_classes for _ in range(pred_boxes.tensor.shape[0])])
                proposal.gt_ious = np.array([0. for _ in range(pred_boxes.tensor.shape[0])])
                proposal.gt_boxes = np.array([[0, 0, 0, 0] for _ in range(pred_boxes.tensor.shape[0])])
            else:
                gt_ious, gt_classes_idx = pairwise_iou(pred_boxes, gt_boxes).max(dim=1)
                gt_classes = input_instance.gt_classes[gt_classes_idx]
                proposal.gt_fg_classes = copy.deepcopy(gt_classes.numpy())
                gt_classes[gt_ious <= self.iou_threshold[0]] = self.num_classes  # background classes
                gt_classes[(self.iou_threshold[0] < gt_ious) & (gt_ious <= self.iou_threshold[1])] = -1  # ignore
                proposal.gt_classes = gt_classes.numpy()
                proposal.gt_ious = gt_ious.numpy()
                proposal.gt_boxes = input_instance.gt_boxes[gt_classes_idx].tensor.numpy()

        return proposal

    def process(self, inputs, outputs):
        self.fg_proposal_list.append(self.process_type(inputs, outputs, "instances"))
        self.bg_proposal_list.append(self.process_type(inputs, outputs, "background"))

    def evaluate(self):
        return self.fg_proposal_list, self.bg_proposal_list


class Proposal:
    """
    A data structure that stores the proposals for a single image.

    Args:
        image_id (str): unique image identifier
        filename (str): image filename
        pred_boxes (numpy.ndarray): predicted boxes
        pred_classes (numpy.ndarray): predicted classes
        pred_scores (numpy.ndarray): class confidence score
        gt_classes (numpy.ndarray, optional): ground-truth classes, including background classes
        gt_boxes (numpy.ndarray, optional): ground-truth boxes
        gt_ious (numpy.ndarray, optional): IoU between predicted boxes and ground-truth boxes
        gt_fg_classes (numpy.ndarray, optional): ground-truth foreground classes, not including background classes

    """
    def __init__(self, image_id, filename, pred_boxes, pred_classes, pred_scores,
                 gt_classes=None, gt_boxes=None, gt_ious=None, gt_fg_classes=None):
        self.image_id = image_id
        self.filename = filename
        self.pred_boxes = pred_boxes
        self.pred_classes = pred_classes
        self.pred_scores = pred_scores
        self.gt_classes = gt_classes
        self.gt_boxes = gt_boxes
        self.gt_ious = gt_ious
        self.gt_fg_classes = gt_fg_classes

    def to_dict(self):
        return {
            "__proposal__": True,
            "image_id": self.image_id,
            "filename": self.filename,
            "pred_boxes": self.pred_boxes.tolist(),
            "pred_classes": self.pred_classes.tolist(),
            "pred_scores": self.pred_scores.tolist(),
            "gt_classes": self.gt_classes.tolist(),
            "gt_boxes": self.gt_boxes.tolist(),
            "gt_ious": self.gt_ious.tolist(),
            "gt_fg_classes": self.gt_fg_classes.tolist()
        }

    def __str__(self):
        pp = pprint.PrettyPrinter(indent=2)
        return pp.pformat(self.to_dict())

    def __len__(self):
        return len(self.pred_boxes)

    def __getitem__(self, item):
        return Proposal(
            image_id=self.image_id,
            filename=self.filename,
            pred_boxes=self.pred_boxes[item],
            pred_classes=self.pred_classes[item],
            pred_scores=self.pred_scores[item],
            gt_classes=self.gt_classes[item],
            gt_boxes=self.gt_boxes[item],
            gt_ious=self.gt_ious[item],
            gt_fg_classes=self.gt_fg_classes[item]
        )


class ProposalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Proposal):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


def asProposal(dict):
    if '__proposal__' in dict:
        return Proposal(
            dict["image_id"],
            dict["filename"],
            np.array(dict["pred_boxes"]),
            np.array(dict["pred_classes"]),
            np.array(dict["pred_scores"]),
            np.array(dict["gt_classes"]),
            np.array(dict["gt_boxes"]),
            np.array(dict["gt_ious"]),
            np.array(dict["gt_fg_classes"])
        )
    return dict


class PersistentProposalList(list):
    """
    A data structure that stores the proposals for a dataset.

    Args:
        filename (str, optional): filename indicating where to cache
    """
    def __init__(self, filename=None):
        super(PersistentProposalList, self).__init__()
        self.filename = filename

    def load(self):
        """
        Load from cache.

        Return:
            whether succeed
        """
        if os.path.exists(self.filename):
            print("Reading from cache: {}".format(self.filename))
            with open(self.filename, "r") as f:
                self.extend(json.load(f, object_hook=asProposal))
            return True
        else:
            return False

    def flush(self):
        """
        Flush to cache.
        """
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, "w") as f:
            json.dump(self, f, cls=ProposalEncoder)
        print("Write to cache: {}".format(self.filename))


def flatten(proposal_list, max_number=10000):
    """
    Flatten a list of proposals

    Args:
        proposal_list (list):  a list of proposals grouped by images
        max_number (int): maximum number of kept proposals for each image

    """
    flattened_list = []
    for proposals in proposal_list:
        for i in range(min(len(proposals), max_number)):
            flattened_list.append(proposals[i:i+1])
    return flattened_list


class ProposalDataset(datasets.VisionDataset):
    """
    A dataset for proposals.

    Args:
        proposal_list (list): list of Proposal
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        crop_func: (ExpandCrop, optional):
    """
    def __init__(self, proposal_list: List[Proposal], transform: Optional[Callable] = None, crop_func=None):
        super(ProposalDataset, self).__init__("", transform=transform)
        self.proposal_list = list(filter(lambda p: len(p) > 0, proposal_list))  # remove images without proposals
        self.loader = default_loader
        self.crop_func = crop_func

    def __getitem__(self, index: int):
        # get proposals for the index-th image
        proposals = self.proposal_list[index]
        img = self.loader(proposals.filename)

        # random sample a proposal
        proposal = proposals[random.randint(0, len(proposals)-1)]
        image_width, image_height = img.width, img.height
        # proposal_dict = proposal.to_dict()
        # proposal_dict.update(width=img.width, height=img.height)

        # crop the proposal from the whole image
        x1, y1, x2, y2 = proposal.pred_boxes
        top, left, height, width = int(y1), int(x1), int(y2 - y1), int(x2 - x1)
        if self.crop_func is not None:
            top, left, height, width = self.crop_func(img, top, left, height, width)
        img = crop(img, top, left, height, width)

        if self.transform is not None:
            img = self.transform(img)

        return img, {
            "image_id": proposal.image_id,
            "filename": proposal.filename,
            "pred_boxes": proposal.pred_boxes.astype(np.float),
            "pred_classes": proposal.pred_classes.astype(np.long),
            "pred_scores": proposal.pred_scores.astype(np.float),
            "gt_classes": proposal.gt_classes.astype(np.long),
            "gt_boxes": proposal.gt_boxes.astype(np.float),
            "gt_ious": proposal.gt_ious.astype(np.float),
            "gt_fg_classes": proposal.gt_fg_classes.astype(np.long),
            "width": image_width,
            "height": image_height
        }

    def __len__(self):
        return len(self.proposal_list)


class ExpandCrop:
    """
    The input of the bounding box adaptor (the crops of objects) will be larger than the original
    predicted box, so that the bounding box adapter could access more location information.
    """
    def __init__(self, expand=1.):
        self.expand = expand

    def __call__(self, img, top, left, height, width):
        cx = left + width / 2.
        cy = top + height / 2.
        height = round(height * self.expand)
        width = round(width * self.expand)
        new_top = round(cy - height / 2.)
        new_left = round(cx - width / 2.)
        return new_top, new_left, height, width