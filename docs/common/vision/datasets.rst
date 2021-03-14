Vision Datasets
=============================


Unsupervised DA for Classification
---------------------------------------------------------


--------------------------------------
ImageList
--------------------------------------

.. autoclass:: common.vision.datasets.imagelist.ImageList
   :members:

-------------------------------------
Office-31
-------------------------------------

.. autoclass:: common.vision.datasets.office31.Office31
   :members:
   :inherited-members:

---------------------------------------
Office-Caltech
---------------------------------------

.. autoclass:: common.vision.datasets.officecaltech.OfficeCaltech
   :members:
   :inherited-members:

---------------------------------------
Office-Home
---------------------------------------

.. autoclass:: common.vision.datasets.officehome.OfficeHome
   :members:
   :inherited-members:

--------------------------------------
VisDA-2017
--------------------------------------

.. autoclass:: common.vision.datasets.visda2017.VisDA2017
   :members:
   :inherited-members:

--------------------------------------
DomainNet
--------------------------------------

.. autoclass:: common.vision.datasets.domainnet.DomainNet
   :members:
   :inherited-members:

--------------------------------------
MNIST
--------------------------------------

.. autoclass:: common.vision.datasets.digits.MNIST
   :members:


--------------------------------------
USPS
--------------------------------------

.. autoclass:: common.vision.datasets.digits.USPS
   :members:


--------------------------------------
SVHN
--------------------------------------

.. autoclass:: common.vision.datasets.digits.SVHN
   :members:


Partial DA for Classification
----------------------------------------------------

---------------------------------------
Partial Wrapper
---------------------------------------

.. autofunction:: common.vision.datasets.partial.partial

.. autofunction:: common.vision.datasets.partial.default_partial


---------------------------------------
Caltech-256->ImageNet-1k
---------------------------------------

.. autoclass:: common.vision.datasets.partial.caltech_imagenet.CaltechImageNet
   :members:


---------------------------------------
ImageNet-1k->Caltech-256
---------------------------------------

.. autoclass:: common.vision.datasets.partial.imagenet_caltech.ImageNetCaltech
   :members:


Open Set DA for Classification
------------------------------------------------------

---------------------------------------
Open Set Wrapper
---------------------------------------

.. autofunction:: common.vision.datasets.openset.open_set

.. autofunction:: common.vision.datasets.openset.default_open_set


Unsupervised DA for Regression
------------------------------------------------------

---------------------------------------
ImageRegression
---------------------------------------

.. autoclass:: common.vision.datasets.regression.image_regression.ImageRegression
   :members:

---------------------------------------
DSprites
---------------------------------------
.. autoclass:: common.vision.datasets.regression.dsprites.DSprites
   :members:

---------------------------------------
MPI3D
---------------------------------------
.. autoclass:: common.vision.datasets.regression.mpi3d.MPI3D
   :members:


Unsupervised DA for Segmentation
-----------------------------------------------

---------------------------------------
SegmentationList
---------------------------------------
.. autoclass:: common.vision.datasets.segmentation.segmentation_list.SegmentationList
   :members:

---------------------------------------
Cityscapes
---------------------------------------
.. autoclass:: common.vision.datasets.segmentation.cityscapes.Cityscapes

---------------------------------------
GTA5
---------------------------------------
.. autoclass:: common.vision.datasets.segmentation.gta5.GTA5

---------------------------------------
Synthia
---------------------------------------
.. autoclass:: common.vision.datasets.segmentation.synthia.Synthia


---------------------------------------
Foggy Cityscapes
---------------------------------------
.. autoclass:: common.vision.datasets.segmentation.cityscapes.FoggyCityscapes


Unsupervised DA for Keypoint Detection
-----------------------------------------------

---------------------------------------
Dataset Base for Keypoint Detection
---------------------------------------
.. autoclass:: common.vision.datasets.keypoint_detection.keypoint_dataset.KeypointDataset
   :members:

.. autoclass:: common.vision.datasets.keypoint_detection.keypoint_dataset.Body16KeypointDataset
   :members:

.. autoclass:: common.vision.datasets.keypoint_detection.keypoint_dataset.Hand21KeypointDataset
   :members:

---------------------------------------
Rendered Handpose Dataset
---------------------------------------
.. autoclass:: common.vision.datasets.keypoint_detection.rendered_hand_pose.RenderedHandPose
   :members:

---------------------------------------
Hand-3d-Studio Dataset
---------------------------------------
.. autoclass:: common.vision.datasets.keypoint_detection.hand_3d_studio.Hand3DStudio
   :members:

---------------------------------------
FreiHAND Dataset
---------------------------------------
.. autoclass:: common.vision.datasets.keypoint_detection.freihand.FreiHand
   :members:

---------------------------------------
Surreal Dataset
---------------------------------------
.. autoclass:: common.vision.datasets.keypoint_detection.surreal.SURREAL
   :members:

---------------------------------------
LSP Dataset
---------------------------------------
.. autoclass:: common.vision.datasets.keypoint_detection.lsp.LSP
   :members:

---------------------------------------
Human3.6M Dataset
---------------------------------------
.. autoclass:: common.vision.datasets.keypoint_detection.human36m.Human36M
   :members:

