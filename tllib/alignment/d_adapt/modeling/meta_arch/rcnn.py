"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
from typing import Optional, Callable, Tuple, Any, List, Sequence, Dict
import numpy as np

from detectron2.utils.events import get_event_storage
from detectron2.structures import Instances
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from tllib.vision.models.object_detection.meta_arch import TLGeneralizedRCNN


@META_ARCH_REGISTRY.register()
class DecoupledGeneralizedRCNN(TLGeneralizedRCNN):
    """
    Generalized R-CNN for Decoupled Adaptation (D-adapt).
    Similar to that in in Supervised Learning, DecoupledGeneralizedRCNN has the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction

    Different from that in Supervised Learning, DecoupledGeneralizedRCNN
    1. accepts unlabeled images and uses the feedbacks from adaptors as supervision during training
    2. generate foreground and background proposals during inference

    Args:
        backbone: a backbone module, must follow detectron2's backbone interface
        proposal_generator: a module that generates proposals using backbone features
        roi_heads: a ROI head that performs per-region computation
        pixel_mean, pixel_std: list or tuple with #channels element,
            representing the per-channel mean and std to be used to normalize
            the input image
        input_format: describe the meaning of channels of input. Needed by visualization
        vis_period: the period to run visualization. Set to 0 to disable.
        finetune (bool): whether finetune the detector or train from scratch. Default: True

    Inputs:
        - batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
          Each item in the list contains the inputs for one image.
          For now, each item in the list is a dict that contains:
            * image: Tensor, image in (C, H, W) format.
            * instances (optional): groundtruth :class:`Instances`
            * feedbacks (optional): :class:`Instances`, feedbacks from adaptors.
            * "height", "width" (int): the output resolution of the model, used in inference.
              See :meth:`postprocess` for details.
        - labeled (bool, optional): whether has ground-truth label

    Outputs:
        - outputs (during inference): A list of dict where each dict is the output for one input image.
          The dict contains a key "instances" whose value is a :class:`Instances`.
          The :class:`Instances` object has the following keys:
          "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        - losses (during training): A dict of different losses
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], labeled=True):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if "feedbacks" in batched_inputs[0]:
            feedbacks = [x["feedbacks"].to(self.device) for x in batched_inputs]
        else:
            feedbacks = None

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, labeled=labeled)
        _, _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, feedbacks, labeled=labeled)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals, feedbacks)

        return losses

    def visualize_training(self, batched_inputs, proposals, feedbacks=None):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
            feedbacks (list): a list that contains feedbacks from adaptors. Both
                batched_inputs and feedbacks should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()

            num_classes = self.roi_heads.box_predictor.num_classes
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
            break  # only visualize one image in a batch

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)

        results, background_results, _ = self.roi_heads(images, features, proposals, None)
        processed_results = []
        for results_per_image, background_results_per_image, input_per_image, image_size in zip(
                results, background_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            background_r = detector_postprocess(background_results_per_image, height, width)
            processed_results.append({"instances": r, 'background': background_r})
        return processed_results
