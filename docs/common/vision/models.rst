Vision Models
===========================

ResNets
---------------------------------

.. automodule:: common.vision.models.resnet
   :members:

Models for Digits Dataset
--------------------------

.. automodule:: common.vision.models.digits
   :members:

Segmentation Models
----------------------------------

.. autofunction:: common.vision.models.segmentation.deeplabv2.deeplabv2_resnet101


Keypoint Detection Models
----------------------------------

.. autofunction:: common.vision.models.keypoint_detection.pose_resnet.pose_resnet101

.. autoclass:: common.vision.models.keypoint_detection.pose_resnet.PoseResNet

.. autoclass:: common.vision.models.keypoint_detection.pose_resnet.Upsampling


Keypoint Detection Loss
----------------------------------

.. autoclass:: common.vision.models.keypoint_detection.loss.JointsMSELoss

.. autoclass:: common.vision.models.keypoint_detection.loss.JointsKLLoss


ReID Models
-----------------------------------
.. autoclass:: common.vision.models.reid.resnet.ReidResNet

.. automodule:: common.vision.models.reid.resnet
    :members:

.. autoclass:: common.vision.models.reid.identifier.ReIdentifier
    :members:

ReID Loss
-----------------------------------
.. autoclass:: common.vision.models.reid.loss.TripletLoss

ReID Sampler
-----------------------------------
.. autoclass:: common.utils.data.RandomMultipleGallerySampler
