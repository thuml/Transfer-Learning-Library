Datasets
=============================

Cross-Domain Classification
---------------------------------------------------------


--------------------------------------
ImageList
--------------------------------------

.. autoclass:: tllib.vision.datasets.imagelist.ImageList
   :members:

-------------------------------------
Office-31
-------------------------------------

.. autoclass:: tllib.vision.datasets.office31.Office31
   :members:
   :inherited-members:

---------------------------------------
Office-Caltech
---------------------------------------

.. autoclass:: tllib.vision.datasets.officecaltech.OfficeCaltech
   :members:
   :inherited-members:

---------------------------------------
Office-Home
---------------------------------------

.. autoclass:: tllib.vision.datasets.officehome.OfficeHome
   :members:
   :inherited-members:

--------------------------------------
VisDA-2017
--------------------------------------

.. autoclass:: tllib.vision.datasets.visda2017.VisDA2017
   :members:
   :inherited-members:

--------------------------------------
DomainNet
--------------------------------------

.. autoclass:: tllib.vision.datasets.domainnet.DomainNet
   :members:
   :inherited-members:

--------------------------------------
PACS
--------------------------------------

.. autoclass:: tllib.vision.datasets.pacs.PACS
   :members:


--------------------------------------
MNIST
--------------------------------------

.. autoclass:: tllib.vision.datasets.digits.MNIST
   :members:


--------------------------------------
USPS
--------------------------------------

.. autoclass:: tllib.vision.datasets.digits.USPS
   :members:


--------------------------------------
SVHN
--------------------------------------

.. autoclass:: tllib.vision.datasets.digits.SVHN
   :members:


Partial Cross-Domain Classification
----------------------------------------------------

---------------------------------------
Partial Wrapper
---------------------------------------

.. autofunction:: tllib.vision.datasets.partial.partial

.. autofunction:: tllib.vision.datasets.partial.default_partial


---------------------------------------
Caltech-256->ImageNet-1k
---------------------------------------

.. autoclass:: tllib.vision.datasets.partial.caltech_imagenet.CaltechImageNet
   :members:


---------------------------------------
ImageNet-1k->Caltech-256
---------------------------------------

.. autoclass:: tllib.vision.datasets.partial.imagenet_caltech.ImageNetCaltech
   :members:


Open Set Cross-Domain Classification
------------------------------------------------------

---------------------------------------
Open Set Wrapper
---------------------------------------

.. autofunction:: tllib.vision.datasets.openset.open_set

.. autofunction:: tllib.vision.datasets.openset.default_open_set


Cross-Domain Regression
------------------------------------------------------

---------------------------------------
ImageRegression
---------------------------------------

.. autoclass:: tllib.vision.datasets.regression.image_regression.ImageRegression
   :members:

---------------------------------------
DSprites
---------------------------------------
.. autoclass:: tllib.vision.datasets.regression.dsprites.DSprites
   :members:

---------------------------------------
MPI3D
---------------------------------------
.. autoclass:: tllib.vision.datasets.regression.mpi3d.MPI3D
   :members:


Cross-Domain Segmentation
-----------------------------------------------

---------------------------------------
SegmentationList
---------------------------------------
.. autoclass:: tllib.vision.datasets.segmentation.segmentation_list.SegmentationList
   :members:

---------------------------------------
Cityscapes
---------------------------------------
.. autoclass:: tllib.vision.datasets.segmentation.cityscapes.Cityscapes

---------------------------------------
GTA5
---------------------------------------
.. autoclass:: tllib.vision.datasets.segmentation.gta5.GTA5

---------------------------------------
Synthia
---------------------------------------
.. autoclass:: tllib.vision.datasets.segmentation.synthia.Synthia


---------------------------------------
Foggy Cityscapes
---------------------------------------
.. autoclass:: tllib.vision.datasets.segmentation.cityscapes.FoggyCityscapes


Cross-Domain Keypoint Detection
-----------------------------------------------

---------------------------------------
Dataset Base for Keypoint Detection
---------------------------------------
.. autoclass:: tllib.vision.datasets.keypoint_detection.keypoint_dataset.KeypointDataset
   :members:

.. autoclass:: tllib.vision.datasets.keypoint_detection.keypoint_dataset.Body16KeypointDataset
   :members:

.. autoclass:: tllib.vision.datasets.keypoint_detection.keypoint_dataset.Hand21KeypointDataset
   :members:

---------------------------------------
Rendered Handpose Dataset
---------------------------------------
.. autoclass:: tllib.vision.datasets.keypoint_detection.rendered_hand_pose.RenderedHandPose
   :members:

---------------------------------------
Hand-3d-Studio Dataset
---------------------------------------
.. autoclass:: tllib.vision.datasets.keypoint_detection.hand_3d_studio.Hand3DStudio
   :members:

---------------------------------------
FreiHAND Dataset
---------------------------------------
.. autoclass:: tllib.vision.datasets.keypoint_detection.freihand.FreiHand
   :members:

---------------------------------------
Surreal Dataset
---------------------------------------
.. autoclass:: tllib.vision.datasets.keypoint_detection.surreal.SURREAL
   :members:

---------------------------------------
LSP Dataset
---------------------------------------
.. autoclass:: tllib.vision.datasets.keypoint_detection.lsp.LSP
   :members:

---------------------------------------
Human3.6M Dataset
---------------------------------------
.. autoclass:: tllib.vision.datasets.keypoint_detection.human36m.Human36M
   :members:

Cross-Domain ReID
------------------------------------------------------

---------------------------------------
Market1501
---------------------------------------

.. autoclass:: tllib.vision.datasets.reid.market1501.Market1501
   :members:

---------------------------------------
DukeMTMC-reID
---------------------------------------

.. autoclass:: tllib.vision.datasets.reid.dukemtmc.DukeMTMC
   :members:

---------------------------------------
MSMT17
---------------------------------------

.. autoclass:: tllib.vision.datasets.reid.msmt17.MSMT17
   :members:


Natural Object Recognition
---------------------------------------------------------


-------------------------------------
Stanford Dogs
-------------------------------------

.. autoclass:: tllib.vision.datasets.stanford_dogs.StanfordDogs
   :members:

-------------------------------------
Stanford Cars
-------------------------------------

.. autoclass:: tllib.vision.datasets.stanford_cars.StanfordCars
   :members:

-------------------------------------
CUB-200-2011
-------------------------------------

.. autoclass:: tllib.vision.datasets.cub200.CUB200
   :members:

-------------------------------------
FVGC Aircraft
-------------------------------------

.. autoclass:: tllib.vision.datasets.aircrafts.Aircraft
   :members:

-------------------------------------
Oxford-IIIT Pets
-------------------------------------

.. autoclass:: tllib.vision.datasets.oxfordpets.OxfordIIITPets
   :members:

-------------------------------------
COCO-70
-------------------------------------

.. autoclass:: tllib.vision.datasets.coco70.COCO70
   :members:

-------------------------------------
DTD
-------------------------------------

.. autoclass:: tllib.vision.datasets.dtd.DTD
   :members:

-------------------------------------
OxfordFlowers102
-------------------------------------

.. autoclass:: tllib.vision.datasets.oxfordflowers.OxfordFlowers102
   :members:

-------------------------------------
Caltech101
-------------------------------------

.. autoclass:: tllib.vision.datasets.caltech101.Caltech101
   :members:


Specialized Image Classification
--------------------------------

-------------------------------------
PatchCamelyon
-------------------------------------

.. autoclass:: tllib.vision.datasets.patchcamelyon.PatchCamelyon
   :members:

-------------------------------------
Retinopathy
-------------------------------------

.. autoclass:: tllib.vision.datasets.retinopathy.Retinopathy
   :members:

-------------------------------------
EuroSAT
-------------------------------------

.. autoclass:: tllib.vision.datasets.eurosat.EuroSAT
   :members:

-------------------------------------
Resisc45
-------------------------------------

.. autoclass:: tllib.vision.datasets.resisc45.Resisc45
   :members:

-------------------------------------
Food-101
-------------------------------------

.. autoclass:: tllib.vision.datasets.food101.Food101
   :members:

-------------------------------------
SUN397
-------------------------------------

.. autoclass:: tllib.vision.datasets.sun397.SUN397
   :members:
