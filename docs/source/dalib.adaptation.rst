DALIB Algorithms
===================================

The adaptation subpackage contains definitions for the following domain adaptation algorithms:

-  `DANN`_
-  `DAN`_
-  `JAN`_
-  `CDAN`_
-  `MCD`_
-  `AFN`_
-  `MDD`_

Besides specific algorithms, this package also provides a recommended image classifier for each algorithms.

We provide benchmarks of different domain adaptation algorithms on *Office-31*, *Office-Home* and *VisDA-2017* as follows.
Note that `Origin` means the accuracy reported by the original paper, while `Avg` is the accuracy reported by DALIB.

*Office-31* accuracy on ResNet-50

===========     ======  ======  ======  ======  ======  ======  ======  ======
Methods         Origin  Avg     A → W   D → W   W → D   A → D   D → A   W → A
Source Only     76.1    79.9    77.0    96.4    99.2    79.1    64.2    63.5
DANN            82.2    85.5    89.1    97.9    100.0   84.1    72.9    68.9
DAN             80.4    83.2    84.0    98.2    100.0   86.1    66.4    64.6
JAN             84.3    86.8    91.8    98.2    100.0   88.0    71.2    71.4
CDAN            87.7    88.3    93.1    98.7    100.0    90.4    75.8    71.9
AFN             85.7    86.3    88.8    98.0    100.0    92.0    70.0    68.9
MDD             88.9    89.2    94.1    98.5    100.0   93.0    76.4    73.0
===========     ======  ======  ======  ======  ======  ======  ======  ======


*Office-Home* accuracy on ResNet-50

=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin  Avg     Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 46.1    58.4    41.1    65.9    73.7    53.1    60.1    63.3    52.2    36.7    71.8    64.8    42.6    75.2
DANN        57.6    65.2    53.8    62.6    74.0    55.8    67.3    67.3    55.8    55.1    77.9    71.1    60.7    81.1
DAN         56.3    61.4    45.6    67.7    73.9    57.7    63.8    66.0    54.9    40.0    74.5    66.2    49.1    77.9
JAN         58.3    65.9    50.8    71.9    76.5    60.6    68.3    68.7    60.5    49.6    76.9    71.0    55.9    80.5
CDAN        65.8    68.8    55.2    72.4    77.6    62.0    69.7    70.9    62.4    54.3    80.5    75.5    61.0    83.8
AFN         67.3    67.7    53.0    71.9    76.8    64.9    70.2    72.0    63.6    50.7    77.7    72.3    56.9    82.1
MDD         68.1    69.6    56.4    75.3    78.4    63.2    73.1    73.3    63.9    54.8    79.7    73.2    60.7    83.7
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
AFN         /           70.7        76.1            74.4
MDD         74.6        74.9        /               78.5
=========== ==========  ==========  ==========  ==========

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
----

.. autoclass:: dalib.adaptation.dan.MultipleKernelMaximumMeanDiscrepancy
    :show-inheritance:


JAN
---------------------------
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


AFN
----------------------------

.. autoclass:: dalib.adaptation.afn.StepwiseAdaptiveFeatureNorm
    :show-inheritance:

.. autoclass:: dalib.adaptation.afn.L2PreservedDropout
    :show-inheritance:

MDD
----------------------------

.. autoclass:: dalib.adaptation.mdd.MarginDisparityDiscrepancy
    :show-inheritance:

.. autoclass:: dalib.adaptation.mdd.ImageClassifier
    :show-inheritance:

.. autofunction:: dalib.adaptation.mdd.shift_log
