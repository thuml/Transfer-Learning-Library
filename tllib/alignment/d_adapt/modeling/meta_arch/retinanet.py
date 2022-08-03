"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, Callable, Tuple, Any, List, Sequence, Dict
import random
import numpy as np

import torch
from torch import Tensor
from detectron2.structures import BoxMode, Boxes, Instances, pairwise_iou, ImageList
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple
from detectron2.modeling import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.events import get_event_storage

from tllib.vision.models.object_detection.meta_arch import TLRetinaNet
from ..matcher import MaxOverlapMatcher


@META_ARCH_REGISTRY.register()
class DecoupledRetinaNet(TLRetinaNet):
    """
    RetinaNet for Decoupled Adaptation (D-adapt).

    Different from that in Supervised Learning, DecoupledRetinaNet
    1. accepts unlabeled images and uses the feedbacks from adaptors as supervision during training
    2. generate foreground and background proposals during inference

    Args:
        backbone: a backbone module, must follow detectron2's backbone interface
        head (nn.Module): a module that predicts logits and regression deltas
            for each level from a list of per-level features
        head_in_features (Tuple[str]): Names of the input feature maps to be used in head
        anchor_generator (nn.Module): a module that creates anchors from a
            list of features. Usually an instance of :class:`AnchorGenerator`
        box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
            instance boxes
        anchor_matcher (Matcher): label the anchors by matching them with ground truth.
        num_classes (int): number of classes. Used to label background proposals.

        # Loss parameters:
        focal_loss_alpha (float): focal_loss_alpha
        focal_loss_gamma (float): focal_loss_gamma
        smooth_l1_beta (float): smooth_l1_beta
        box_reg_loss_type (str): Options are "smooth_l1", "giou"

        # Inference parameters:
        test_score_thresh (float): Inference cls score threshold, only anchors with
            score > INFERENCE_TH are considered for inference (to improve speed)
        test_topk_candidates (int): Select topk candidates before NMS
        test_nms_thresh (float): Overlap threshold used for non-maximum suppression
            (suppress boxes with IoU >= this threshold)
        max_detections_per_image (int):
            Maximum number of detections to return per image during inference
            (100 is based on the limit established for the COCO dataset).

        # Input parameters
        pixel_mean (Tuple[float]):
            Values to be used for image normalization (BGR order).
            To train on images of different number of channels, set different mean & std.
            Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
        pixel_std (Tuple[float]):
            When using pre-trained models in Detectron1 or any MSRA models,
            std has been absorbed into its conv1 weights, so the std needs to be set 1.
            Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
        vis_period (int):
            The period (in terms of steps) for minibatch visualization at train time.
            Set to 0 to disable.
        input_format (str): Whether the model needs RGB, YUV, HSV etc.
        finetune (bool): whether finetune the detector or train from scratch. Default: True

    Inputs:
        - batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
          Each item in the list contains the inputs for one image.
          For now, each item in the list is a dict that contains:
            * image: Tensor, image in (C, H, W) format.
            * instances (optional): groundtruth :class:`Instances`
            * "height", "width" (int): the output resolution of the model, used in inference.
              See :meth:`postprocess` for details.
        - labeled (bool, optional): whether has ground-truth label

    Outputs:
        - outputs: A list of dict where each dict is the output for one input image.
          The dict contains a key "instances" whose value is a :class:`Instances`
          and a key "features" whose value is the features of middle layers.
          The :class:`Instances` object has the following keys:
          "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        - losses: A dict of different losses
    """
    def __init__(self, *args, max_samples_per_level=25, **kwargs):
        super(DecoupledRetinaNet, self).__init__(*args, **kwargs)
        self.max_samples_per_level = max_samples_per_level
        self.max_matcher = MaxOverlapMatcher()

    def forward_training(self, images, features, predictions, gt_instances=None, feedbacks=None, labeled=True):
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4]
        )
        anchors = self.anchor_generator(features)
        if labeled:
            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)
        else:
            proposal_labels, proposal_boxes = self.label_pseudo_anchors(anchors, feedbacks)
            losses = self.losses(anchors, pred_logits, proposal_labels, pred_anchor_deltas, proposal_boxes)
            losses.pop('loss_box_reg')
        return losses

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]], labeled=True):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            if "feedbacks" in batched_inputs[0]:
                feedbacks = [x["feedbacks"].to(self.device) for x in batched_inputs]
            else:
                feedbacks = None

            losses = self.forward_training(images, features, predictions, gt_instances, feedbacks, labeled)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.forward_inference(images, features, predictions)
                    self.visualize_training(batched_inputs, results, feedbacks)

            return losses
        else:
            # sample_background must be called before inference
            # since inference will change predictions
            background_results = self.sample_background(images, features, predictions)
            results = self.forward_inference(images, features, predictions)

            processed_results = []
            for results_per_image, background_results_per_image, input_per_image, image_size in zip(
                    results, background_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                background_r = detector_postprocess(background_results_per_image, height, width)
                processed_results.append({"instances": r, "background": background_r})
            return processed_results

    @torch.no_grad()
    def label_pseudo_anchors(self, anchors, instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            list[Tensor]:
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """
        anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.max_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def sample_background(
        self, images: ImageList, features: List[Tensor], predictions: List[List[Tensor]]
    ):
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4]
        )
        anchors = self.anchor_generator(features)

        results: List[Instances] = []
        for img_idx, image_size in enumerate(images.image_sizes):
            scores_per_image = [x[img_idx].sigmoid() for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.sample_background_single_image(
                anchors, scores_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)
        return results

    def sample_background_single_image(
            self,
            anchors: List[Boxes],
            box_cls: List[Tensor],
            box_delta: List[Tensor],
            image_size: Tuple[int, int],
    ):
        boxes_all = []
        scores_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.max(dim=1).values

            # 1. Keep boxes with confidence score lower than threshold
            keep_idxs = predicted_prob < self.test_score_thresh
            anchor_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Random sample boxes
            anchor_idxs = anchor_idxs[
                random.sample(range(len(anchor_idxs)), k=min(len(anchor_idxs), self.max_samples_per_level))]
            predicted_prob = predicted_prob[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            boxes_all.append(anchors_i.tensor)
            scores_all.append(predicted_prob)

        boxes_all, scores_all = [
            cat(x) for x in [boxes_all, scores_all]
        ]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all)
        result.scores = 1. - scores_all  # the confidence score to be background
        result.pred_classes = torch.tensor([self.num_classes for _ in range(len(scores_all))])
        return result

    def visualize_training(self, batched_inputs, results, feedbacks=None):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements returned by forward_inference().
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()

        num_classes = self.num_classes
        if feedbacks is not None:
            v_feedback_gt = Visualizer(img, None)
            instance = feedbacks[0].to(torch.device("cpu"))
            v_feedback_gt = v_feedback_gt.overlay_instances(
                boxes=instance.proposal_boxes[instance.gt_classes != num_classes])
            feedback_gt_img = v_feedback_gt.get_image()

            v_feedback_gf = Visualizer(img, None)
            v_feedback_gf = v_feedback_gf.overlay_instances(
                boxes=instance.proposal_boxes[instance.gt_classes == num_classes])
            feedback_gf_img = v_feedback_gf.get_image()

            vis_img = np.vstack((anno_img, prop_img, feedback_gt_img, feedback_gf_img))
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = f"Top: GT; Middle: Pred; Bottom: Feedback GT, Feedback GF"
        else:
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"

        storage.put_image(vis_name, vis_img)
