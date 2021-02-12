Vision Datasets
=============================


Unsupervised DA for Classification
---------------------------------------------------------


--------------------------------------
ImageList
--------------------------------------

.. autoclass:: dalib.vision.datasets.imagelist.ImageList
   :members:

-------------------------------------
Office-31
-------------------------------------

.. autoclass:: dalib.vision.datasets.office31.Office31
   :members:
   :inherited-members:

---------------------------------------
Office-Caltech
---------------------------------------

.. autoclass:: dalib.vision.datasets.officecaltech.OfficeCaltech
   :members:
   :inherited-members:

---------------------------------------
Office-Home
---------------------------------------

.. autoclass:: dalib.vision.datasets.officehome.OfficeHome
   :members:
   :inherited-members:

--------------------------------------
VisDA-2017
--------------------------------------

.. autoclass:: dalib.vision.datasets.visda2017.VisDA2017
   :members:
   :inherited-members:

--------------------------------------
DomainNet
--------------------------------------

.. autoclass:: dalib.vision.datasets.domainnet.DomainNet
   :members:
   :inherited-members:

--------------------------------------
MNIST
--------------------------------------

.. autoclass:: dalib.vision.datasets.digits.MNIST
   :members:


--------------------------------------
USPS
--------------------------------------

.. autoclass:: dalib.vision.datasets.digits.USPS
   :members:


--------------------------------------
SVHN
--------------------------------------

.. autoclass:: dalib.vision.datasets.digits.SVHN
   :members:


Partial DA for Classification
----------------------------------------------------

---------------------------------------
Partial Wrapper
---------------------------------------

.. autofunction:: dalib.vision.datasets.partial.partial

.. autofunction:: dalib.vision.datasets.partial.default_partial


---------------------------------------
Caltech-256->ImageNet-1k
---------------------------------------

.. autoclass:: dalib.vision.datasets.partial.caltech_imagenet.CaltechImageNet
   :members:


---------------------------------------
ImageNet-1k->Caltech-256
---------------------------------------

.. autoclass:: dalib.vision.datasets.partial.imagenet_caltech.ImageNetCaltech
   :members:


Open Set DA for Classification
------------------------------------------------------

---------------------------------------
Open Set Wrapper
---------------------------------------

.. autofunction:: dalib.vision.datasets.openset.open_set

.. autofunction:: dalib.vision.datasets.openset.default_open_set


Unsupervised DA for Regression
------------------------------------------------------

---------------------------------------
ImageRegression
---------------------------------------

.. autoclass:: dalib.vision.datasets.regression.image_regression.ImageRegression
   :members:

---------------------------------------
DSprites
---------------------------------------
.. autoclass:: dalib.vision.datasets.regression.dsprites.DSprites
   :members:

---------------------------------------
MPI3D
---------------------------------------
.. autoclass:: dalib.vision.datasets.regression.mpi3d.MPI3D
   :members:


Unsupervised DA for Segmentation
-----------------------------------------------

---------------------------------------
SegmentationList
---------------------------------------
.. autoclass:: dalib.vision.datasets.segmentation.segmentation_list.SegmentationList
   :members:

---------------------------------------
Cityscapes
---------------------------------------
.. autoclass:: dalib.vision.datasets.segmentation.cityscapes.Cityscapes

---------------------------------------
GTA5
---------------------------------------
.. autoclass:: dalib.vision.datasets.segmentation.gta5.GTA5

---------------------------------------
Synthia
---------------------------------------
.. autoclass:: dalib.vision.datasets.segmentation.synthia.Synthia


