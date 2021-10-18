Vision Datasets
=============================

Cross-Domain Classification
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
PACS
--------------------------------------

.. autoclass:: common.vision.datasets.pacs.PACS
   :members:


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


Partial Cross-Domain Classification
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


Open Set Cross-Domain Classification
------------------------------------------------------

---------------------------------------
Open Set Wrapper
---------------------------------------

.. autofunction:: common.vision.datasets.openset.open_set

.. autofunction:: common.vision.datasets.openset.default_open_set


Cross-Domain Regression
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


Cross-Domain Segmentation
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


Cross-Domain Keypoint Detection
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

Cross-Domain ReID
------------------------------------------------------

---------------------------------------
Market1501
---------------------------------------

.. autoclass:: common.vision.datasets.reid.market1501.Market1501
   :members:

---------------------------------------
DukeMTMC-reID
---------------------------------------

.. autoclass:: common.vision.datasets.reid.dukemtmc.DukeMTMC
   :members:

---------------------------------------
MSMT17
---------------------------------------

.. autoclass:: common.vision.datasets.reid.msmt17.MSMT17
   :members:


Natural Object Recognition
---------------------------------------------------------


-------------------------------------
Stanford Dogs
-------------------------------------

.. autoclass:: common.vision.datasets.stanford_dogs.StanfordDogs
   :members:

-------------------------------------
Stanford Cars
-------------------------------------

.. autoclass:: common.vision.datasets.stanford_cars.StanfordCars
   :members:

-------------------------------------
CUB-200-2011
-------------------------------------

.. autoclass:: common.vision.datasets.cub200.CUB200
   :members:

-------------------------------------
FVGC Aircraft
-------------------------------------

.. autoclass:: common.vision.datasets.aircrafts.Aircraft
   :members:

-------------------------------------
Oxford-IIIT Pet
-------------------------------------

.. autoclass:: common.vision.datasets.oxfordpet.OxfordIIITPet
   :members:

-------------------------------------
COCO-70
-------------------------------------

.. autoclass:: common.vision.datasets.coco70.COCO70
   :members:

-------------------------------------
DTD
-------------------------------------

.. autoclass:: common.vision.datasets.dtd.DTD
   :members:

-------------------------------------
OxfordFlowers102
-------------------------------------

.. autoclass:: common.vision.datasets.oxfordflowers.OxfordFlowers102
   :members:


Specialized Image Classification
--------------------------------

-------------------------------------
PatchCamelyon
-------------------------------------

.. autoclass:: common.vision.datasets.patchcamelyon.PatchCamelyon
   :members:

-------------------------------------
Retinopathy
-------------------------------------

.. autoclass:: common.vision.datasets.retinopathy.Retinopathy
   :members:

-------------------------------------
EuroSAT
-------------------------------------

.. autoclass:: common.vision.datasets.eurosat.EuroSAT
   :members:

-------------------------------------
Resisc45
-------------------------------------

.. autoclass:: common.vision.datasets.resisc45.Resisc45
   :members: