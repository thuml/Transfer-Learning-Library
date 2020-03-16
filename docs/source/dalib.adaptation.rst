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

*Office-31* error rates on ResNet-50

=======     ======  ======  ======  ======  ======  ======  ======
Methods     Avg     A → W   D → W   W → D   A → D   D → A   W → A
=======     ======  ======  ======  ======  ======  ======  ======
DANN
DAN
JAN
CDAN
MCD
AFN
MDD
=======     ======  ======  ======  ======  ======  ======  ======

*Office-Home* error rates on ResNet-50

=======     ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     Avg     Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
=======     ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
DANN
DAN
JAN
CDAN
MCD
AFN
MDD
=======     ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

*VisDA-2017* error rates on ResNet-50 and ResNet-101

======= =========   ==========
Methods ResNet-50   ResNet-101
======= =========   ==========
DANN
DAN
JAN
CDAN
MCD
AFN
MDD
======= =========   ==========

.. _DANN: https://arxiv.org/abs/1505.07818
.. _DAN: https://arxiv.org/pdf/1502.02791
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

.. autoclass:: dalib.adaptation.dann.DomainDiscriminator
    :show-inheritance:


DAN
----

.. autoclass:: dalib.adaptation.mmd.MultipleKernelMaximumMeanDiscrepancy
    :show-inheritance:


JAN
---------------------------
.. autoclass:: dalib.adaptation.mmd.JointMultipleKernelMaximumMeanDiscrepancy
    :show-inheritance:


CDAN
----------------------------

.. autoclass:: dalib.adaptation.cdan.ConditionalDomainAdversarialLoss
    :show-inheritance:

.. autoclass:: dalib.adaptation.cdan.DomainDiscriminator
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
