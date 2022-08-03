"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import itertools
import numpy as np
import copy
import logging
from typing import List, Optional, Union
import torch

from detectron2.config import configurable
from detectron2.structures import BoxMode, Boxes, Instances
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.build import filter_images_with_only_crowd_annotations, filter_images_with_few_keypoints, \
    print_instances_class_histogram
from detectron2.data.detection_utils import check_metadata_consistency
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils

from .proposal import Proposal


def load_feedbacks_into_dataset(dataset_dicts, proposals_list: List[Proposal]):
    """
    Load precomputed object feedbacks into the dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposals_list (list[Proposal]): list of Proposal.

    Returns:
        list[dict]: the same format as dataset_dicts, but added feedback field.
    """
    feedbacks = {}

    for record in dataset_dicts:
        image_id = str(record["image_id"])
        feedbacks[image_id] = {
            'pred_boxes': [],
            'pred_classes': [],
        }

    for proposals in proposals_list:
        image_id = str(proposals.image_id)
        feedbacks[image_id]['pred_boxes'] += proposals.pred_boxes.tolist()
        feedbacks[image_id]['pred_classes'] += proposals.pred_classes.tolist()

    # Assuming default bbox_mode of precomputed feedbacks are 'XYXY_ABS'
    bbox_mode = BoxMode.XYXY_ABS

    dataset_dicts_with_feedbacks = []
    for record in dataset_dicts:
        # Get the index of the feedback
        image_id = str(record["image_id"])
        record["feedback_proposal_boxes"] = feedbacks[image_id]["pred_boxes"]
        record["feedback_gt_classes"] = feedbacks[image_id]["pred_classes"]
        record["feedback_gt_boxes"] = feedbacks[image_id]["pred_boxes"]
        record["feedback_bbox_mode"] = bbox_mode
        if sum(map(lambda x: x >= 0, feedbacks[image_id]["pred_classes"])) > 0:  # remove images without feedbacks
            dataset_dicts_with_feedbacks.append(record)

    return dataset_dicts_with_feedbacks


def get_detection_dataset_dicts(names, filter_empty=True, min_keypoints=0, proposals_list=None):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposals_list (optional, list[Proposal]): list of Proposal.


    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    if proposals_list is not None:
        # load precomputed feedbacks for each proposals
        dataset_dicts = load_feedbacks_into_dataset(dataset_dicts, proposals_list)

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if has_instances:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            check_metadata_consistency("thing_classes", names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts


def transform_feedbacks(dataset_dict, image_shape, transforms, *, min_box_size=0):
    """
    Apply transformations to the feedbacks in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        min_box_size (int): proposals with either side smaller than this
            threshold are removed

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "feedback_proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        proposal_boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("feedback_proposal_boxes"),
                dataset_dict.get("feedback_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        proposal_boxes = Boxes(proposal_boxes)
        gt_boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("feedback_gt_boxes"),
                dataset_dict.get("feedback_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        gt_boxes = Boxes(gt_boxes)
        gt_classes = torch.as_tensor(
            dataset_dict.pop("feedback_gt_classes")
        )

        proposal_boxes.clip(image_shape)
        gt_boxes.clip(image_shape)
        keep = proposal_boxes.nonempty(threshold=min_box_size) & (gt_classes >= 0)
        # keep = boxes.nonempty(threshold=min_box_size)
        proposal_boxes = proposal_boxes[keep]
        gt_boxes = gt_boxes[keep]
        gt_classes = gt_classes[keep]

        feedbacks = Instances(image_shape)
        feedbacks.proposal_boxes = proposal_boxes
        feedbacks.gt_boxes = gt_boxes
        feedbacks.gt_classes = gt_classes
        dataset_dict["feedbacks"] = feedbacks


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

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

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        transform_feedbacks(
            dataset_dict, image_shape, transforms
        )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
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