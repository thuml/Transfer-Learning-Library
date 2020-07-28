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

Office-31 accuracy on ResNet-50
---------------------------------

**Note**  We conduct 21 class classification experiments in this setting (follows `OSBP`_) while some paper conducted 11 class classifications.
Thus the accuracies are not always comparable.

===========     ================    ======  ======  ======  ======  ======  ======
Methods         H-score Avg         A → W   D → W   W → D   A → D   D → A   W → A
Source Only     75.9                69.1    86.4    91.2    73.0    68.7    67.2
DANN            80.0                79.0    91.3    93.9    80.5    65.9    69.3
OSBP            87.0                90.4    94.7    97.4    88.8    72.6    78.0
===========     ================    ======  ======  ======  ======  ======  ======

===========     ================    ======  ======  ======  ======  ======  ======
Methods         OS Avg              A → W   D → W   W → D   A → D   D → A   W → A
Source Only     72.2                63.1    90.1    95.1    68.2    59.5    56.9
DANN            79.7                82.7    94.9    91.9    83.8    58.8    65.9
OSBP            86.1                89.8    94.4    99.0    87.4    72.5    73.6
===========     ================    ======  ======  ======  ======  ======  ======

===========     ================    ======  ======  ======  ======  ======  ======
Methods         OS* Avg             A → W   D → W   W → D   A → D   D → A   W → A
Source Only     71.6                62.4    90.4    95.4    67.6    58.3    55.5
DANN            79.6                83.1    95.3    91.6    84.1    57.9    65.5
OSBP            86.0                89.7    94.3    99.1    87.3    72.5    73.1
===========     ================    ======  ======  ======  ======  ======  ======

===========     ================    ======  ======  ======  ======  ======  ======
Methods         UNK Avg             A → W   D → W   W → D   A → D   D → A   W → A
Source Only     82.6                77.5    82.8    87.2    79.3    83.7    85.2
DANN            81.0                75.3    87.6    96.3    77.1    76.4    73.5
OSBP            88.1                91.0    95.1    95.7    90.4    72.6    83.6
===========     ================    ======  ======  ======  ======  ======  ======

Office-Home accuracy on ResNet-50
-----------------------------------

=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     H-score Avg         Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 59.8                52.3    66.0    71.8    52.2    59.1    63.6    56.8    43.7    68.1    64.3    50.3    69.2
DANN        64.2                58.1    66.3    73.3    60.0    64.3    68.0    60.1    54.5    69.3    64.9    59.6    71.8
OSBP        68.5                62.9    69.6    75.8    66.0    68.9    72.0    64.6    58.2    75.7    71.0    60.9    76.1
=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     OS Avg              Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 52.9                43.1    61.8    68.7    40.4    50.0    56.0    47.6    32.9    67.4    61.4    40.6    64.7
DANN        59.2                47.4    60.8    67.5    54.2    60.2    64.9    56.4    47.5    67.3    57.1    53.7    73.5
OSBP        68.7                60.1    76.3    81.4    60.9    68.9    73.2    63.6    51.6    76.7    70.1    60.0    81.5
=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     OS* Avg             Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 52.1                42.0    61.4    68.4    38.8    49.1    55.3    46.6    31.4    67.3    61.1    39.5    64.3
DANN        58.7                46.1    60.3    67.0    53.7    59.8    64.7    56.1    46.8    67.1    56.3    53.1    73.6
OSBP        68.7                59.9    76.8    81.8    60.4    69.0    73.3    63.5    50.9    76.8    70.0    59.9    81.9
=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     UNK Avg             Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 72.5                69.3    71.4    75.5    79.7    74.2    74.9    72.5    71.9    68.9    67.7    69.1    74.8
DANN        71.5                78.6    73.7    80.9    67.8    69.5    71.6    64.7    65.2    71.6    76.6    67.7    70.0
OSBP        68.8                66.3    63.5    70.7    72.8    68.8    70.7    65.6    68.0    74.6    72.0    61.9    71.0
=========== ================    ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

VisDA-2017 accuracy on ResNet-50
-----------------------------------

=========== ========    ======  =====   ====
Methods     H-score     OS      OS*     UNK
Source Only 43.2        41.2    54.9    47.1
DANN        53.0        49.4    75.0    59.6
OSBP        68.7        64.8    92.4    76.1
=========== ========    ======  =====   ====


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
