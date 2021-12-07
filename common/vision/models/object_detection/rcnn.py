"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN as GeneralizedRCNNBase


class GeneralizedRCNN(GeneralizedRCNNBase):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, *args, finetune=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetune = finetune

    def get_parameters(self, lr=1.):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        return [
            (self.backbone, 0.1 * lr if self.finetune else lr),
            (self.proposal_generator, lr),
            (self.roi_heads, lr),
        ]