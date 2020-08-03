DALIB Algorithms
===================================

The adaptation subpackage contains definitions for the following domain adaptation algorithms:

-  `DANN`_
-  `DAN`_
-  `JAN`_
-  `CDAN`_
-  `MCD`_
-  `MDD`_

Besides specific algorithms, this package also provides a recommended image classifier for each algorithms.

We provide benchmarks of different domain adaptation algorithms on *Office-31*, *Office-Home* and *VisDA-2017* as follows.
Note that `Origin` means the accuracy reported by the original paper, while `Avg` is the accuracy reported by DALIB.

*Office-31* accuracy on ResNet-50

===========     ======  ======  ======  ======  ======  ======  ======  ======
Methods         Origin  Avg     A → W   D → W   W → D   A → D   D → A   W → A
Source Only     76.1    79.5    75.8    95.5    99.0    79.3    63.6    63.8
DANN            82.2    86.4    91.7    97.9    100.0   82.9    72.8    73.3
DAN             80.4    83.7    84.2    98.4    100.0   87.3    66.9    65.2
JAN             84.3    87.3    93.7    98.4    100.0   89.4    71.2    71.0
CDAN            87.7    88.7    93.1    98.6    100.0   93.4    75.6    71.5
MDD             88.9    89.2    93.6    98.6    100.0   93.6    76.7    72.9
===========     ======  ======  ======  ======  ======  ======  ======  ======


*Office-Home* accuracy on ResNet-50

=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin  Avg     Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 46.1    58.2    41.5    65.8    73.6    52.2    59.5    63.6    51.5    36.4    71.3    65.2    42.8    75.4
DANN        57.6    65.5    52.7    61.8    73.4    57.4    67.2    69.6    57.2    55.4    79.0    71.4    60.0    81.1
DAN         56.3    61.6    45.5    67.9    73.9    57.6    63.7    66.2    55.2    39.7    74.3    66.8    49.1    78.7
JAN         58.3    65.9    50.4    71.8    76.7    60.0    67.7    68.9    60.4    49.8    77.0    71.2    55.6    81.0
CDAN        65.8    68.8    54.4    70.9    77.9    61.6    71.1    71.9    62.3    54.9    80.7    75.1    60.8    83.7
MDD         68.1    69.5    56.2    74.9    78.8    63.4    72.5    72.6    63.8    54.6    80.0    73.5    60.1    83.7
=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

*VisDA-2017* accuracy on ResNet-50 and ResNet-101

=========== ==========  ==========  ==========  ==========
Methods     Origin      DALIB       Origin      DALIB
Backbone    ResNet-50   ResNet-50   ResNet-101  ResNet-101
Source Only /           55.1        52.4        58.3
DANN        /           72.6        57.4            72.9
DAN         /           60.6        61.1            64.8
JAN         61.6        64.9        /               68.0
CDAN        66.8        74.6        /               74.5
MCD         69.2        69.1        71.9            77.3
MDD         74.6        74.9        /               78.5
=========== ==========  ==========  ==========  ==========

*DomainNet* accuracy on ResNet-101

=========== ======  ======  ======  ======  ======  ======
Source Only clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    18.0    32.7    50.6    39.4    35.2
inf         35.7    N/A	    31.1    50.0    26.5    35.8
pnt         41.1    17.8    N/A     56.8    35.0    37.7
real        48.6    22.9    48.8    N/A	    36.1	39.1
skt         49.0    15.3    34.8    46.1    N/A     36.3
Avg         43.6    18.5    36.9    50.9    34.3    36.8
=========== ======  ======  ======  ======  ======  ======

=========== ======  ======  ======  ======  ======  ======
DANN        clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    19.7    35.4    53.9    44.2    38.3
inf         26.7    N/A     23.8    28.8    23.7    25.8
pnt         37.2    18.7    N/A     51.1    36.0    35.8
real        50.6    22.1    47.9    N/A     39.0    39.9
skt         54.0    19.7    42.7    52.8    N/A     42.3
Avg         42.1    20.1    37.5    46.7    35.7    36.4
=========== ======  ======  ======  ======  ======  ======

=========== ======  ======  ======  ======  ======  ======
DAN         clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    17.3    37.9    54.0    42.6    38.0
inf         34.9    N/A	    33.4    46.5    29.9    36.2
pnt         43.9    17.7    N/A     55.9    39.3    39.2
real        50.1    20.0    48.6    N/A	    38.4	39.3
skt         54.2    17.5    44.2    53.4    N/A     42.3
Avg         45.8    18.1    41.0    52.5    37.6    39.0
=========== ======  ======  ======  ======  ======  ======

=========== ======  ======  ======  ======  ======  ======
CDAN        clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    20.8    40.0    56.1    45.5    40.6
inf         31.2    N/A	    30.0    41.4    24.7    31.8
pnt         44.6    20.5    N/A     57.0    39.9    40.5
real        55.3    24.1    52.6    N/A	    42.4	43.6
skt         56.7    21.3    46.2    55.0    N/A     44.8
Avg         47.0    21.7    42.2    52.4    38.1    40.3
=========== ======  ======  ======  ======  ======  ======

=========== ======  ======  ======  ======  ======  ======
MDD         clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    21.2    42.9    59.5    47.5    42.8
inf         35.3    N/A	    34.0    49.6    29.4    37.1
pnt         48.6    19.7    N/A     59.4    42.6    42.6
real        58.3    24.9    53.7    N/A	    46.2	45.8
skt         58.7    20.7    46.5    57.7    N/A     45.9
Avg         50.2    21.6    44.3    56.6    41.4    42.8
=========== ======  ======  ======  ======  ======  ======

.. _DANN: https://arxiv.org/abs/1505.07818
.. _DAN: https://arxiv.org/abs/1502.02791
.. _JAN: https://arxiv.org/abs/1605.06636
.. _CDAN: https://arxiv.org/abs/1705.10667
.. _MCD: https://arxiv.org/abs/1712.02560
.. _AFN: https://arxiv.xilesou.top/abs/1811.07456
.. _MDD: https://arxiv.org/abs/1904.05801

.. currentmodule:: dalib.adaptation


DANN
----------------------------
.. autoclass:: dalib.adaptation.dann.DomainAdversarialLoss
    :show-inheritance:

DAN
----------------------------
.. autoclass:: dalib.adaptation.dan.MultipleKernelMaximumMeanDiscrepancy
    :show-inheritance:

JAN
----------------------------
.. autoclass:: dalib.adaptation.jan.JointMultipleKernelMaximumMeanDiscrepancy
    :show-inheritance:

CDAN
----------------------------
.. autoclass:: dalib.adaptation.cdan.ConditionalDomainAdversarialLoss
    :show-inheritance:

.. autoclass:: dalib.adaptation.cdan.RandomizedMultiLinearMap
    :show-inheritance:

.. autoclass:: dalib.adaptation.cdan.MultiLinearMap
    :show-inheritance:

MCD
----------------------------
.. autofunction:: dalib.adaptation.mcd.classifier_discrepancy

.. autofunction:: dalib.adaptation.mcd.entropy

.. autoclass:: dalib.adaptation.mcd.ImageClassifierHead
    :show-inheritance:

MDD
----------------------------
.. autoclass:: dalib.adaptation.mdd.MarginDisparityDiscrepancy
    :show-inheritance:

.. autoclass:: dalib.adaptation.mdd.ImageClassifier
    :show-inheritance:

.. autofunction:: dalib.adaptation.mdd.shift_log
