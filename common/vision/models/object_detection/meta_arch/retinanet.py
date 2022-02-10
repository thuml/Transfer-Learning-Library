"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

from detectron2.modeling.meta_arch.retinanet import RetinaNet as RetinaNetBase
from detectron2.modeling import detector_postprocess


class TLRetinaNet(RetinaNetBase):
    """
    RetinaNet for Transfer Learning.

    Different from that in Supervised Learning, TLRetinaNet
    1. accepts unlabeled images during training (return no losses)
    2. return both detection outputs, features, and losses during training

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
    def __init__(self, *args, finetune=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetune = finetune

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]], labeled=True):
        """"""
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)

        if self.training:
            if labeled:
                assert not torch.jit.is_scripting(), "Not supported"
                assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                losses = self.forward_training(images, features, predictions, gt_instances)
            else:
                losses = {}
            outputs = {"features": features}
            return outputs, losses
        else:
            results = self.forward_inference(images, features, predictions)
            if torch.jit.is_scripting():
                return results

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def get_parameters(self, lr=1.):
        """Return a parameter list which decides optimization hyper-parameters,
            such as the learning rate of each layer
        """
        return [
            (self.backbone.bottom_up, 0.1 * lr if self.finetune else lr),
            (self.backbone.fpn_lateral4, lr),
            (self.backbone.fpn_output4, lr),
            (self.backbone.fpn_lateral5, lr),
            (self.backbone.fpn_output5, lr),
            (self.backbone.top_block, lr),
            (self.head, lr),
        ]