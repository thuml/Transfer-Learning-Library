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
DANN            43.4    83.6    64.1	96.9	98.1	72.0	83.1	87.2
PADA            92.7    95.5    93.2	100.0	99.4	89.8	94.6	95.8
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
-----------------------------------

=========== ======= ======= ====    ====
Methods     Origin  Avg     I→C     C→I
Source Only 68.9    73.3    71.8	74.8
DANN        60.8    72.7    71.4	73.9
PADA        72.8    80.1    80.9	79.2
=========== ======= ======= ====    ====


VisDA-2017 accuracy on ResNet-50
-----------------------------------

=========== ======= ======= ====    ====
Methods     Origin  Avg     R→S     S→R
Source Only 54.8    58.7    58.0	59.4
DANN        62.4    54.8    59.0	50.5
PADA        65.0    64.7    69.0	60.4
=========== ======= ======= ====    ====

.. _PADA: https://arxiv.org/abs/1808.04205

.. currentmodule:: dalib.adaptation


-----------
Algorithms
-----------

PADA
----------------------------

.. autoclass:: dalib.adaptation.pada.ClassWeightModule
    :show-inheritance:

.. autoclass:: dalib.adaptation.pada.AutomaticUpdateClassWeightModule
    :show-inheritance:
    :members:

.. autofunction::  dalib.adaptation.pada.collect_classification_results
