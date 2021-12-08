"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
from typing import List
from detectron2.structures import Instances
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    Res5ROIHeads,
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
