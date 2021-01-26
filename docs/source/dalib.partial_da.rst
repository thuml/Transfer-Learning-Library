Partial Domain Adaptation
==========================================

The adaptation subpackage contains definitions for the following domain adaptation algorithms:

-  `PADA`_

-----------
Benchmarks
-----------

We provide benchmarks of different partial domain adaptation algorithms on *Office-31*, *Office-Home*, *ImageNet-Caltech* and *VisDA-2017* as follows.
Note that `Origin` means the accuracy reported by the original paper, while `Avg` is the accuracy reported by DALIB.

Office-31 accuracy on ResNet-50
---------------------------------

===========     ======  ======  ======  ======  ======  ======  ======  ======
Methods         Origin  Avg     A → W   D → W   W → D   A → D   D → A   W → A
Source Only     75.6    90.1    78.3	98.3	99.4	87.3	88.5	88.8
DANN            43.4    82.4    60.0	94.9	98.1	71.3	84.9	85.0
PADA            92.7    93.8    86.4	100.0	100.0	87.3	93.8	95.4
===========     ======  ======  ======  ======  ======  ======  ======  ======



Office-Home accuracy on ResNet-50
-----------------------------------

=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin  Avg     Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 53.7    60.1    42.0    66.9    78.5    56.4    55.2    65.4    57.9    36.0    75.5    68.7    43.6    74.8
DANN        47.4    57.0    46.2    59.3    76.9    47.0    47.4    56.4    51.6    38.8    72.1    68.0    46.1    74.2
PADA        62.1    65.9    52.9    69.3    82.8    59.0    57.5    66.4    66.0    41.7    82.5    78.0    50.2    84.1
=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

ImageNet-Caltech accuracy on ResNet-50
--------------------------------------

=========== ======= ======= ====    ====
Methods     Origin  Avg     I→C     C→I
Source Only 68.9    73.3    71.8	74.8
DANN        60.8    73.1    71.6	74.5
PADA        72.8    79.2    79.2	79.1
=========== ======= ======= ====    ====


VisDA-2017 accuracy on ResNet-50
-----------------------------------

Note that `Origin` means the accuracy reported by the original paper,
`Mean` refers to the accuracy average over classes, while `Avg` refers to accuracy average over samples.

=========== ==========  ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin      Mean    plane   bcycl   bus     car     horse   knife   Avg
Source Only 45.3        50.9	59.2	31.3	68.7	73.2	69.3	3.4	    60.0
DANN        51.0        55.9	88.4	34.1	72.1	50.7	61.9	27.8	57.1
Source Only 53.5        60.5	89.4	35.1	72.5	69.2	86.7	10.1	66.8
=========== ==========  ======= ======= ======= ======= ======= ======= ======= =======


.. _PADA: https://arxiv.org/abs/1808.04205

.. currentmodule:: dalib.adaptation


-----------
Algorithms
-----------

PADA
----------------------------

.. autoclass:: dalib.adaptation.pada.ClassWeightModule

.. autoclass:: dalib.adaptation.pada.AutomaticUpdateClassWeightModule
    :members:

.. autofunction::  dalib.adaptation.pada.collect_classification_results
