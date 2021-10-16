===============================
Image Classification
===============================

We provide benchmarks of different domain generalization algorithms on `PACS`_, `Office-Home`_,
`iWildCam-Wilds`_, `Camelyon17-Wilds`_, `FMoW-Wilds`_.
Those domain generalization algorithms includes:

- :ref:`IBN`
- :ref:`MIXSTYLE`
- :ref:`MLDG`
- :ref:`IRM`
- :ref:`VREX`
- :ref:`GroupDRO`
- :ref:`CORAL`

.. note::

    `DomainBed <https://github.com/facebookresearch/DomainBed>`_ proposed three model selection methods, our hyper
    parameter is selected based on model's performance on `training-domain validation set` (first rule proposed).
    Concretely, we select model with highest accuracy on `training-domain validation set` during whole training
    process and use selected checkpoint to test on target domain.

.. note::

    Different from `DomainBed <https://github.com/facebookresearch/DomainBed>`_, we do not freeze `BatchNorm2d` layers
    and do not insert additional `Dropout` layer except for `PACS` dataset. Besides, we use `SGD` with momentum as
    default optimizer and find it usually achieves better results than `Adam`. During training, a cosine learning rate
    decay strategy is used.

.. note::
    - ``ERM`` refers to the model trained with `ERM <https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034>`_, which is a strong baseline.
    - ``Avg`` is the average accuracy.
    - ``Acc1`` is the top-1 accuracy on `OOD` test set for Wilds datasets.

.. _PACS:

-----------------------------------
PACS accuracy on ResNet-50
-----------------------------------

======== ===== ===== ===== ===== =====
Methods   avg    A     C     P     S
ERM      86.4  88.5  78.4  97.2  81.4
IBN      87.8  88.2  84.5  97.1  81.4
MixStyle 87.4  87.8  82.3  95.0  84.5
MLDG     87.2  88.2  81.4  96.6  82.5
IRM      86.9  88.0  82.5  98.0  79.0
VREx     87.0  87.2  82.3  97.4  81.0
GroupDRO 87.3  88.9  81.7  97.8  80.8
CORAL    86.4  89.1  80.0  97.4  79.1
======== ===== ===== ===== ===== =====

.. _Office-Home:

-----------------------------------
Office-Home accuracy on ResNet-50
-----------------------------------

======== ===== ===== ===== ===== =====
Methods   avg    A     C     P     R
ERM      70.8  68.3  55.9  78.9  80.0
IBN      69.9  67.4  55.2  77.3  79.6
MixStyle 71.7  66.8  58.1  78.0  79.9
MLDG     70.3  65.9  57.6  78.2  79.6
IRM      70.3  66.7  54.8  78.6  80.9
VREx     70.2  66.9  54.9  78.2  80.9
GroupDRO 70.0  66.7  55.2  78.8  79.9
CORAL    70.9  68.3  55.4  78.8  81.0
======== ===== ===== ===== ===== =====

.. _iWildCam-Wilds:

----------------------------------------
iWildCam-Wilds accuracy on ResNet-50
----------------------------------------

======== ======
Methods   acc1
ERM       75.4
IBN       77.3
MixStyle  71.0
IRM       75.5
VREx      71.5
GroupDRO  28.0
CORAL     71.0
======== ======

.. _Camelyon17-Wilds:

----------------------------------------
Camelyon17-Wilds accuracy on ResNet-50
----------------------------------------

======== ======
Methods   acc1
ERM       94.6
IBN       96.1
MixStyle  94.2
MLDG      91.2
IRM       94.9
VREx      88.2
GroupDRO  93.1
CORAL     90.6
======== ======

.. _FMoW-Wilds:

--------------------------------------
FMoW-Wilds accuracy on DenseNet-121
--------------------------------------

======== ======
Methods   acc1
ERM       53.0
MLDG      47.4
IRM       48.1
VREx      50.4
GroupDRO  47.5
CORAL     50.0
======== ======
