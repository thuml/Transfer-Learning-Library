"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Dict, Optional, List

import torch
from detectron2.structures import ImageList, Instances
from detectron2.modeling.proposal_generator import (
    RPN,
    PROPOSAL_GENERATOR_REGISTRY,
)


@PROPOSAL_GENERATOR_REGISTRY.register()
class TLRPN(RPN):
    """
    Region Proposal Network, introduced by `Faster R-CNN`.

    Args:
        in_features (list[str]): list of names of input features to use
        head (nn.Module): a module that predicts logits and regression deltas
            for each level from a list of per-level features
        anchor_generator (nn.Module): a module that creates anchors from a
            list of features. Usually an instance of :class:`AnchorGenerator`
        anchor_matcher (Matcher): label the anchors by matching them with ground truth.
        box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
            instance boxes
        batch_size_per_image (int): number of anchors per image to sample for training
        positive_fraction (float): fraction of foreground anchors to sample for training
        pre_nms_topk (tuple[float]): (train, test) that represents the
            number of top k proposals to select before NMS, in
            training and testing.
        post_nms_topk (tuple[float]): (train, test) that represents the
            number of top k proposals to select after NMS, in
            training and testing.
        nms_thresh (float): NMS threshold used to de-duplicate the predicted proposals
        min_box_size (float): remove proposal boxes with any side smaller than this threshold,
            in the unit of input image pixels
        anchor_boundary_thresh (float): legacy option
        loss_weight (float|dict): weights to use for losses. Can be single float for weighting
            all rpn losses together, or a dict of individual weightings. Valid dict keys are:
                "loss_rpn_cls" - applied to classification loss
                "loss_rpn_loc" - applied to box regression loss
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"

    Inputs:
        - images (ImageList): input images of length `N`
        - features (dict[str, Tensor]): input data as a mapping from feature
          map name to tensor. Axis 0 represents the number of images `N` in
          the input data; axes 1-3 are channels, height, and width, which may
          vary between feature maps (e.g., if a feature pyramid is used).
        - gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
          Each `Instances` stores ground-truth instances for the corresponding image.
        - labeled (bool, optional): whether has ground-truth label. Default: True

    Outputs:
        - proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
        - loss: dict[Tensor] or None
    """
    def __init__(self, *args, **kwargs):
        super(TLRPN, self).__init__(*args, **kwargs)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
        labeled: Optional[bool] = True
    ):
        features = [features[f] for f in self.in_features]
        # print(torch.max(features[0]))
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training and labeled:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

