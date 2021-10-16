Vision Models
===========================

------------------------------
Image Classification
------------------------------

ResNets
---------------------------------

.. automodule:: common.vision.models.resnet
   :members:

LeNet
--------------------------

.. automodule:: common.vision.models.digits.lenet
   :members:

DTN
--------------------------

.. automodule:: common.vision.models.digits.dtn
   :members:


----------------------------------
Semantic Segmentation
----------------------------------

.. autofunction:: common.vision.models.segmentation.deeplabv2.deeplabv2_resnet101


----------------------------------
Keypoint Detection
----------------------------------

PoseResNet
--------------------------

.. autofunction:: common.vision.models.keypoint_detection.pose_resnet.pose_resnet101

.. autoclass:: common.vision.models.keypoint_detection.pose_resnet.PoseResNet

.. autoclass:: common.vision.models.keypoint_detection.pose_resnet.Upsampling


Joint Loss
----------------------------------

.. autoclass:: common.vision.models.keypoint_detection.loss.JointsMSELoss

.. autoclass:: common.vision.models.keypoint_detection.loss.JointsKLLoss


-----------------------------------
Re-Identification
-----------------------------------

Models
---------------
.. autoclass:: common.vision.models.reid.resnet.ReidResNet

.. automodule:: common.vision.models.reid.resnet
    :members:

.. autoclass:: common.vision.models.reid.identifier.ReIdentifier
    :members:

Loss
-----------------------------------
.. autoclass:: common.vision.models.reid.loss.TripletLoss

Sampler
-----------------------------------
.. autoclass:: common.utils.data.RandomMultipleGallerySampler
