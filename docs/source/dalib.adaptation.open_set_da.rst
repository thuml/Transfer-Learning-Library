Open Set Domain Adaptation
==========================================

The adaptation subpackage contains definitions for the following domain adaptation algorithms:

-  `OSBP`_

-----------
Benchmarks
-----------

We provide benchmarks of different open set domain adaptation algorithms on *Office-31*, *Office-Home* and *VisDA-2017* as follows.
Note that `OS` means normalized accuracy for all classes including the unknown as one class, `OS*` means normalized accuracy only on
known classes and `UNK` is the accuracy of unknown samples. However, in `OS`, the accuracy of each common class has the same contribution
as the whole `unknown` class. Thus we use ``H-score`` proposed by `CMU`_ to better measure the abilities of different open set domain adaptation algorithms.

.. math::
    \textit{H-score} = 2 \cdot \dfrac{ \textit{OS*} \cdot \textit{UNK} }{ \textit{OS*} + \textit{UNK} }

The new evaluation metric is high only when both the `OS*` and `UNK` are high.

**Note** We report the best h-score in all epochs.
DANN (baseline model) will degrade performance as training progresses, thus the final h-score will be much lower than reported.
In contrast, OSBP will improve performance stably.

Office-31 H-Score on ResNet-50
---------------------------------

**Note** We conduct 21 class classification experiments in this setting (follows `OSBP`_).

===========     ================    ======  ======  ======  ======  ======  ======
Methods         Avg                 A → W   D → W   W → D   A → D   D → A   W → A
Source Only     75.9                67.7    85.7    91.4    72.1    68.4    67.8
DANN            80.4                81.4    89.1    92.0    82.5    66.7    70.4
OSBP            87.8                90.7    96.4    97.5    88.7    77.0    76.7
===========     ================    ======  ======  ======  ======  ======  ======

Office-Home H-score on ResNet-50
-----------------------------------

=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     Avg                 Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 59.8                55.2	65.2	71.4	52.8	59.6	65.2	55.8	44.8	68.0	63.8	49.4	68.0
DANN        64.8                55.2	65.2	71.4	52.8	59.6	65.2	55.8	44.8	68.0	63.8	49.4	68.0
OSBP        68.6                62.0	70.8	76.5	66.4	68.8	73.8	65.8	57.1	75.4	70.6	60.6	75.9
=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

VisDA-2017 performance on ResNet-50
-----------------------------------

=========== ========    ======  =====   ====    ======= ======= ======= ======= ======= =======
Methods     H-score     OS      OS*     UNK     bcycl   bus     car     mcycl   train   truck
Source Only 42.6        37.6    34.7    55.1    42.6    6.4     30.5    67.1    84.0    0.2
DANN        57.8        50.4    45.6    78.9    20.1	71.4	29.5	74.4	67.8	10.4
OSBP        75.4        67.3    62.9    94.3    63.7	75.9	49.6	74.4	86.2	27.3
=========== ========    ======  =====   ====    ======= ======= ======= ======= ======= =======


.. _OSBP: https://arxiv.org/abs/1804.10427
.. _CMU: http://ise.thss.tsinghua.edu.cn/~mlong/publications.html

.. currentmodule:: dalib.adaptation

-----------
Algorithms
-----------

OSBP
------

.. autoclass:: dalib.adaptation.osbp.UnknownClassBinaryCrossEntropy
    :show-inheritance:
