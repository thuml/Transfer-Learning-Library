Regression Domain Adaptation
===============================================

The adaptation subpackage contains definitions for the following domain adaptation algorithms:

-  `DD`_

-----------
Benchmarks
-----------

We provide benchmarks of different domain adaptation algorithms on *dSprites* as follows.

Note that labels are all normalized to [0, 1] to eliminate the effects of diverse scale in regression values.
We repeat experiments on DD for three times and report the average error of the final epoch.

dSprites error on ResNet-18
---------------------------------
===========     ======  ======  ======  ======  ======  ======  ======
Methods         Avg     C → N   C → S   N → C   N → S   S → C   S → N
Source Only     0.157   0.232   0.271   0.081   0.220   0.038   0.092
DD              0.057   0.047	0.080	0.030	0.095	0.053	0.037
===========     ======  ======  ======  ======  ======  ======  ======

.. _DD: https://arxiv.org/abs/1904.05801

.. currentmodule:: dalib.adaptation

-----------
Algorithms
-----------

DD/MDD
----------------------------
.. autoclass:: dalib.adaptation.mdd.RegressionMarginDisparityDiscrepancy

.. autoclass:: dalib.adaptation.mdd.ImageRegressor

