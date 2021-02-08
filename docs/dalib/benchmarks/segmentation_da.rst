Segmentation Domain Adaptation
==========================================

We provide benchmarks of different segmentation domain adaptation algorithms on `GTA5->Cityscapes`_ and `Synthia->Cityscapes`_ as follows.
Those domain adaptation algorithms includes:

-  :ref:`ADVENT`
-  :ref:`FDA`
-  :ref:`CycleGAN`


.. note::

    - ``Origin`` means the accuracy reported by the original paper.
    - ``mIoU`` is the mean IoU reported by DALIB.
    - ``Src Only`` refers to the model trained with data from the source domain.
    - ``Oracle`` refers to the model trained with data from the target domain.

.. _GTA5->Cityscapes:

GTA5->Cityscapes mIoU on deeplabv2 (ResNet-101)
-----------------------------------------------

=========== ======  ======= ======= ========    ========    ======= ======= ======= ======= ======= =======
Methods     Origin  mIoU    road    sidewalk    building    wall    fence   pole    light   sign    veg
Src Only    27.1    37.3    66.5    17.4        73.3        13.4    21.5    22.8    30.1    17.1    82.2
ADVENT      43.8    43.8    89.3    33.9        80.3        24.0    25.2    27.8    36.7    18.2    84.3
FDA         44.6    45.6    85.5    31.7        81.8        27.1    24.9    28.9    38.1    23.2    83.7
Oracle      65.1    70.5    97.4    79.7        90.1        53.0    50.0    48.0    55.5    67.2    90.2
=========== ======  ======= ======= ========    ========    ======= ======= ======= ======= ======= =======


=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     terrain sky     person  rider   car     truck   bus     train   mbike   bike
Src Only    7.1     73.6    57.4    28.4    78.6    36.1    13.4    1.5     31.9    36.2
ADVENT      33.9    81.3    59.8    28.4    84.3    34.1    44.4    0.1     33.2    12.9
FDA         40.3    80.6    60.5    30.3    79.1    32.8    45.1    5.0	    32.4    35.2
Oracle      60.0    93.0    72.7    55.2    92.7    76.5    78.5    56.0    54.6    68.8
=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

.. _Synthia->Cityscapes:

Synthia->Cityscapes mIoU on deeplabv2 (ResNet-101)
--------------------------------------------------

=========   ======  ====    ====    ========    ========    =====   ====    ====    ====    ======  =====   ====    ====    =====   ====
Methods     Origin  mIoU    road    sidewalk    building    light   sign    veg     sky     person  rider   car     bus     mbike   bike
Src Only    22.1    41.5    59.6    21.1        77.4        7.7     17.6    78.0    84.5    53.2    16.9    65.9    24.9    8.5     24.8
ADVENT      47.6    47.9    88.3    44.9        80.5        4.5     9.1     81.3    86.2    52.9    21.0    82.0    30.3    11.9    30.2
FDA         /       43.9    62.5    23.7        78.5        9.4     15.7    78.3    81.1    52.3    18.7    79.8    32.5    8.7     29.6
Oracle      71.7    76.6    97.4    79.7        90.1        55.5    67.2    90.2    93.0    72.7    55.2    92.7    78.5    54.6    68.8
=========   ======  ====    ====    ========    ========    =====   ====    ====    ====    ======  =====   ====    ====    =====   ====
