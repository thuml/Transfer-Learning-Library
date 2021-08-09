===============================
Classification
===============================

We provide benchmarks of different domain generalization algorithms on `PACS`_, `Office-Home`_, `DomainNet`_.
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

.. _PACS:

-----------------------------------
PACS accuracy on ResNet-50
-----------------------------------

======== ===== ===== ===== ===== =====
Methods   avg    A     C     P     S
ERM      86.2  88.4  79.2  96.8  80.5
IBN      87.0  88.1  84.8  97.6  77.4
MixStyle 87.2  88.8  81.7  96.2  82.2
MLDG     87.2  88.2  81.4  96.6  82.5
IRM      86.9  89.0  80.5  97.5  80.5
VREx     87.5  88.0  79.9  98.2  84.0
GroupDRO 86.8  88.2  81.7  97.5  79.6
CORAL    86.2  88.9  79.8  97.8  78.4
======== ===== ===== ===== ===== =====

.. _Office-Home:

-----------------------------------
Office-Home accuracy on ResNet-50
-----------------------------------

======== ===== ===== ===== ===== =====
Methods   avg    A     C     P     R
ERM      70.6  68.0  55.2  79.3  80.1
IBN      70.2  67.6  55.3  79.9  78.0
MixStyle 71.0  68.3  57.7  78.1  80.0
MLDG     70.3  65.9  57.6  78.2  79.6
IRM      70.4  67.8  54.9  78.1  80.6
VREx     70.5  67.9  54.6  78.5  81.1
GroupDRO 70.8  67.7  57.3  78.3  80.0
CORAL    70.7  67.6  55.0  78.8  81.4
======== ===== ===== ===== ===== =====

.. _DomainNet:

-----------------------------------
DomainNet accuracy on ResNet-50
-----------------------------------

======== ===== ========= =========== ========== =========== ====== ========
Methods   avg   clipart   infograph   painting   quickdraw   real   sketch
ERM      44.0    64.3        21.3       50.8       16.3      57.5    54.2
IBN      43.9    64.8        20.8       50.9       16.3      57.3    53.4
MixStyle 44.6    65.0        21.4       50.3       16.5      59.4    54.7
MLDG
IRM      35.4    51.3        15.2       39.3       14.8      45.6    44.4
VREx     41.4    60.7        20.2       45.8       15.1      56.3    50.3
GroupDRO 35.6    53.3        18.4       37.4       13.3      46.3    44.6
CORAL    44.8    65.1        21.6       50.7       16.6      60.0    54.5
======== ===== ========= =========== ========== =========== ====== ========
