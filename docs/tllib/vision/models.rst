Models
===========================

------------------------------
Image Classification
------------------------------

ResNets
---------------------------------

.. automodule:: tllib.vision.models.resnet
   :members:

LeNet
--------------------------

.. automodule:: tllib.vision.models.digits.lenet
   :members:

DTN
--------------------------

.. automodule:: tllib.vision.models.digits.dtn
   :members:

----------------------------------
Object Detection
----------------------------------

.. autoclass:: tllib.vision.models.object_detection.meta_arch.TLGeneralizedRCNN
   :members:

.. autoclass:: tllib.vision.models.object_detection.meta_arch.TLRetinaNet
   :members:

.. autoclass:: tllib.vision.models.object_detection.proposal_generator.rpn.TLRPN

.. autoclass:: tllib.vision.models.object_detection.roi_heads.TLRes5ROIHeads
    :members:

.. autoclass:: tllib.vision.models.object_detection.roi_heads.TLStandardROIHeads
    :members:

----------------------------------
Semantic Segmentation
----------------------------------

.. autofunction:: tllib.vision.models.segmentation.deeplabv2.deeplabv2_resnet101


----------------------------------
Keypoint Detection
----------------------------------

PoseResNet
--------------------------

.. autofunction:: tllib.vision.models.keypoint_detection.pose_resnet.pose_resnet101

.. autoclass:: tllib.vision.models.keypoint_detection.pose_resnet.PoseResNet

.. autoclass:: tllib.vision.models.keypoint_detection.pose_resnet.Upsampling


Joint Loss
----------------------------------

.. autoclass:: tllib.vision.models.keypoint_detection.loss.JointsMSELoss

.. autoclass:: tllib.vision.models.keypoint_detection.loss.JointsKLLoss


-----------------------------------
Re-Identification
-----------------------------------

Models
---------------
.. autoclass:: tllib.vision.models.reid.resnet.ReidResNet

.. automodule:: tllib.vision.models.reid.resnet
    :members:

.. autoclass:: tllib.vision.models.reid.identifier.ReIdentifier
    :members:

Loss
-----------------------------------
.. autoclass:: tllib.vision.models.reid.loss.TripletLoss

Sampler
-----------------------------------
.. autoclass:: tllib.utils.data.RandomMultipleGallerySampler
