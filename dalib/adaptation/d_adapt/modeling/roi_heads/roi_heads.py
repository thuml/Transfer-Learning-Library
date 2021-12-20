"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import numpy as np
import random
from typing import List, Tuple, Dict

from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    Res5ROIHeads,
    StandardROIHeads
)
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.sampling import subsample_labels
from detectron2.layers import nonzero_tuple


from .fast_rcnn import DecoupledFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class DecoupledRes5ROIHeads(Res5ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.

    It typically contains logic to

      1. when training on labeled source domain, match proposals with ground truth and sample them
      2. when training on unlabeled target domain, match proposals with feedbacks from adaptors and sample them
      3. crop the regions and extract per-region features using proposals
      4. make per-region predictions with different heads
    """
    def __init__(self, *args, **kwargs):
        super(DecoupledRes5ROIHeads, self).__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg, input_shape)
        ret["res5"], out_channels = cls._build_res5_block(cfg)
        box_predictor = DecoupledFastRCNNOutputLayers(cfg, ShapeSpec(channels=out_channels, height=1, width=1))
        ret["box_predictor"] = box_predictor
        return ret

    def forward(self, images, features, proposals, targets=None, feedbacks=None, labeled=True):
        """
        Prepare some proposals to be used to train the ROI heads.
        When training on labeled source domain, it performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        When training on unlabeled target domain, it performs box matching between `proposals` and `feedbacks`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
            feedbacks (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the feedback of per-instance annotations
                for the i-th input image.  Specify `feedbacks` during training only.
                It have the same fields as `targets`.
            labeled (bool, optional): whether has ground-truth label

        Returns:
            tuple[list[Instances], list[Instances], dict]:
                a tuple containing foreground proposals (`Instances`), background proposals (`Instances`) and a dict of different losses.

            Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        del images

        if self.training:
            assert targets
            if labeled:
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.label_and_sample_feedbacks(feedbacks)
            del targets
        proposal_boxes = [x.proposal_boxes for x in proposals]

        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if not labeled:
                losses.pop("loss_box_reg")
            return [], [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            boxes = self.box_predictor.predict_boxes(predictions, proposals)
            scores = self.box_predictor.predict_probs(predictions, proposals)
            image_shapes = [x.image_size for x in proposals]
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_shapes,
                self.box_predictor.test_score_thresh,
                self.box_predictor.test_nms_thresh,
                self.box_predictor.test_topk_per_image,
            )
            background_instances, _ = fast_rcnn_sample_background(
                [box.tensor for box in proposal_boxes],
                scores,
                image_shapes,
                self.box_predictor.test_score_thresh,
                self.box_predictor.test_nms_thresh,
                self.box_predictor.test_topk_per_image,
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            background_instances = self.forward_with_given_boxes(features, background_instances)
            return pred_instances, background_instances, {}

    @torch.no_grad()
    def label_and_sample_feedbacks(
            self, feedbacks, batch_size_per_image=256
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `feedbacks`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            feedbacks (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the feedback of per-instance annotations
                for the i-th input image.  Specify `feedbacks` during training only.
                It have the same fields as `targets`.

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for feedbacks_per_image in feedbacks:
            gt_classes = feedbacks_per_image.gt_classes
            positive = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[0]
            # ensure each batch consists the same number bg and fg boxes
            batch_size = min(batch_size_per_image, max(2 * positive.numel(), 1))
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                gt_classes, batch_size, self.positive_fraction, self.num_classes
            )

            sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
            gt_classes = gt_classes[sampled_idxs]

            # Set target attributes of the sampled proposals:
            proposals_per_image = feedbacks_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head_pseudo/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head_pseudo/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt


@ROI_HEADS_REGISTRY.register()
class DecoupledStandardROIHeads(StandardROIHeads):
    """
    The Standard ROIHeads used by most models, such as FPN and C5.
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    It typically contains logic to

      1. when training on labeled source domain, match proposals with ground truth and sample them
      2. when training on unlabeled target domain, match proposals with feedbacks from adaptors and sample them
      3. crop the regions and extract per-region features using proposals
      4. make per-region predictions with different heads
    """
    def __init__(self, *args, **kwargs):
        super(DecoupledStandardROIHeads, self).__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg, input_shape)
        box_predictor = DecoupledFastRCNNOutputLayers(cfg, ret['box_head'].output_shape)
        ret["box_predictor"] = box_predictor
        return ret

    def forward(self, images, features, proposals, targets=None, feedbacks=None, labeled=True):
        """
        Prepare some proposals to be used to train the ROI heads.
        When training on labeled source domain, it performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        When training on unlabeled target domain, it performs box matching between `proposals` and `feedbacks`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
            feedbacks (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the feedback of per-instance annotations
                for the i-th input image.  Specify `feedbacks` during training only.
                It have the same fields as `targets`.
            labeled (bool, optional): whether has ground-truth label

        Returns:
            tuple[list[Instances], list[Instances], dict]:
                a tuple containing foreground proposals (`Instances`), background proposals (`Instances`) and a dict of different losses.

            Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        del images

        if self.training:
            assert targets
            if labeled:
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.label_and_sample_feedbacks(feedbacks)
            del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))

            if not labeled:
                losses.pop('loss_box_reg')
            return [], [], losses
        else:
            pred_instances, predictions = self._forward_box(features, proposals)
            scores = self.box_predictor.predict_probs(predictions, proposals)
            image_shapes = [x.image_size for x in proposals]
            proposal_boxes = [x.proposal_boxes for x in proposals]
            background_instances, _ = fast_rcnn_sample_background(
                [box.tensor for box in proposal_boxes],
                scores,
                image_shapes,
                self.box_predictor.test_score_thresh,
                self.box_predictor.test_nms_thresh,
                self.box_predictor.test_topk_per_image,
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            background_instances = self.forward_with_given_boxes(features, background_instances)
            return pred_instances, background_instances, {}

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_feedbacks(
            self, feedbacks, batch_size_per_image=256
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for feedbacks_per_image in feedbacks:
            gt_classes = feedbacks_per_image.gt_classes
            positive = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[0]
            # ensure each batch consists the same number bg and fg boxes
            batch_size = min(batch_size_per_image, max(2 * positive.numel(), 1))
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                gt_classes, batch_size, self.positive_fraction, self.num_classes
            )

            sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
            gt_classes = gt_classes[sampled_idxs]

            # Set target attributes of the sampled proposals:
            proposals_per_image = feedbacks_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head_pseudo/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head_pseudo/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt


def fast_rcnn_sample_background(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_sample_background_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the background proposals.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_sample_background_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_sample_background_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image background samples. .

    Args:
        Same as `fast_rcnn_sample_background`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_sample_background`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    num_classes = scores.shape[1]
    # Only keep background proposals
    scores = scores[:, -1:]
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, 1, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS only for background class
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if 0 <= topk_per_image < len(keep):
        idx = list(range(len(keep)))
        idx = random.sample(idx, k=topk_per_image)
        idx = sorted(idx)
        keep = keep[idx]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1] + num_classes - 1
    return result, filter_inds[:, 0]
