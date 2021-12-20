"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Tuple, Dict
import torch
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN as GeneralizedRCNNBase, get_event_storage
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class TLGeneralizedRCNN(GeneralizedRCNNBase):
    """
    Generalized R-CNN for Transfer Learning.
    Similar to that in in Supervised Learning, TLGeneralizedRCNN has the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction

    Different from that in Supervised Learning, TLGeneralizedRCNN
    1. accepts unlabeled images during training (return no losses)
    2. return both detection outputs, features, and losses during training

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
            * proposals (optional): :class:`Instances`, precomputed proposals.
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

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], labeled=True):
        """"""
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0] and labeled:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, labeled)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        outputs, detector_losses = self.roi_heads(images, features, proposals, gt_instances, labeled)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        outputs['features'] = features
        return outputs, losses

    def get_parameters(self, lr=1.):
        """Return a parameter list which decides optimization hyper-parameters,
            such as the learning rate of each layer
        """
        return [
            (self.backbone, 0.1 * lr if self.finetune else lr),
            (self.proposal_generator, lr),
            (self.roi_heads, lr),
        ]