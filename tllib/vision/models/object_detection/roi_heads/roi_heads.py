"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
from typing import List, Dict
from detectron2.structures import Instances
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    Res5ROIHeads,
    StandardROIHeads,
    select_foreground_proposals,
)


@ROI_HEADS_REGISTRY.register()
class TLRes5ROIHeads(Res5ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.

    Args:
        in_features (list[str]): list of backbone feature map names to use for
            feature extraction
        pooler (ROIPooler): pooler to extra region features from backbone
        res5 (nn.Sequential): a CNN to compute per-region features, to be used by
            ``box_predictor`` and ``mask_head``. Typically this is a "res5"
            block from a ResNet.
        box_predictor (nn.Module): make box predictions from the feature.
            Should have the same interface as :class:`FastRCNNOutputLayers`.
        mask_head (nn.Module): transform features to make mask predictions

    Inputs:
        - images (ImageList):
        - features (dict[str,Tensor]): input data as a mapping from feature
          map name to tensor. Axis 0 represents the number of images `N` in
          the input data; axes 1-3 are channels, height, and width, which may
          vary between feature maps (e.g., if a feature pyramid is used).
        - proposals (list[Instances]): length `N` list of `Instances`. The i-th
          `Instances` contains object proposals for the i-th input image,
          with fields "proposal_boxes" and "objectness_logits".
        - targets (list[Instances], optional): length `N` list of `Instances`. The i-th
          `Instances` contains the ground-truth per-instance annotations
          for the i-th input image.  Specify `targets` during training only.
          It may have the following fields:
            - gt_boxes: the bounding box of each instance.
            - gt_classes: the label for each instance with a category ranging in [0, #class].
            - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
            - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        - labeled (bool, optional): whether has ground-truth label. Default: True

    Outputs:
        - list[Instances]: length `N` list of `Instances` containing the
          detected instances. Returned during inference only; may be [] during training.

        - dict[str->Tensor]:
          mapping from a named loss to a tensor storing the loss. Used during training only.
    """
    def __init__(self, *args, **kwargs):
        super(TLRes5ROIHeads, self).__init__(*args, **kwargs)

    def forward(self, images, features, proposals, targets=None, labeled=True):
        """"""
        del images

        if self.training:
            if labeled:
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.sample_unlabeled_proposals(proposals)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            if labeled:
                losses = self.box_predictor.losses(predictions, proposals)
                if self.mask_on:
                    proposals, fg_selection_masks = select_foreground_proposals(
                        proposals, self.num_classes
                    )
                    # Since the ROI feature transform is shared between boxes and masks,
                    # we don't need to recompute features. The mask loss is only defined
                    # on foreground proposals, so we need to select out the foreground
                    # features.
                    mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                    # del box_features
                    losses.update(self.mask_head(mask_features, proposals))
            else:
                losses = {}
            outputs = {
                'predictions': predictions[0],
                'box_features': box_features
            }
            return outputs, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    @torch.no_grad()
    def sample_unlabeled_proposals(
        self, proposals: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some unlabeled proposals.
        It returns top ``self.batch_size_per_image`` samples from proposals

        Args:
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".

        Returns:
            length `N` list of `Instances`s containing the proposals sampled for training.
        """
        return [proposal[:self.batch_size_per_image] for proposal in proposals]


@ROI_HEADS_REGISTRY.register()
class TLStandardROIHeads(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    Args:
        box_in_features (list[str]): list of feature names to use for the box head.
        box_pooler (ROIPooler): pooler to extra region features for box head
        box_head (nn.Module): transform features to make box predictions
        box_predictor (nn.Module): make box predictions from the feature.
            Should have the same interface as :class:`FastRCNNOutputLayers`.
        mask_in_features (list[str]): list of feature names to use for the mask
            pooler or mask head. None if not using mask head.
        mask_pooler (ROIPooler): pooler to extract region features from image features.
            The mask head will then take region features to make predictions.
            If None, the mask head will directly take the dict of image features
            defined by `mask_in_features`
        mask_head (nn.Module): transform features to make mask predictions
        keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
        train_on_pred_boxes (bool): whether to use proposal boxes or
            predicted boxes from the box head to train other heads.

    Inputs:
        - images (ImageList):
        - features (dict[str,Tensor]): input data as a mapping from feature
          map name to tensor. Axis 0 represents the number of images `N` in
          the input data; axes 1-3 are channels, height, and width, which may
          vary between feature maps (e.g., if a feature pyramid is used).
        - proposals (list[Instances]): length `N` list of `Instances`. The i-th
          `Instances` contains object proposals for the i-th input image,
          with fields "proposal_boxes" and "objectness_logits".
        - targets (list[Instances], optional): length `N` list of `Instances`. The i-th
          `Instances` contains the ground-truth per-instance annotations
          for the i-th input image.  Specify `targets` during training only.
          It may have the following fields:
            - gt_boxes: the bounding box of each instance.
            - gt_classes: the label for each instance with a category ranging in [0, #class].
            - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
            - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        - labeled (bool, optional): whether has ground-truth label. Default: True

    Outputs:
        - list[Instances]: length `N` list of `Instances` containing the
          detected instances. Returned during inference only; may be [] during training.

        - dict[str->Tensor]:
          mapping from a named loss to a tensor storing the loss. Used during training only.
    """
    def __init__(self, *args, **kwargs):
        super(TLStandardROIHeads, self).__init__(*args, **kwargs)

    def forward(self, images, features, proposals, targets=None, labeled=True):
        """"""
        del images
        if self.training:
            if labeled:
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.sample_unlabeled_proposals(proposals)
        del targets

        if self.training:
            if labeled:
                outputs, losses = self._forward_box(features, proposals)
                # Usually the original proposals used by the box head are used by the mask, keypoint
                # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
                # predicted by the box head.
                losses.update(self._forward_mask(features, proposals))
                losses.update(self._forward_keypoint(features, proposals))
            else:
                losses = {}
            return outputs, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

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
            outputs = {
                'predictions': predictions[0],
                'box_features': box_features
            }
            return outputs, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    @torch.no_grad()
    def sample_unlabeled_proposals(
        self, proposals: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some unlabeled proposals.
        It returns top ``self.batch_size_per_image`` samples from proposals

        Args:
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".

        Returns:
            length `N` list of `Instances`s containing the proposals sampled for training.
        """
        return [proposal[:self.batch_size_per_image] for proposal in proposals]



