Regression Domain Adaptation
===============================================

The adaptation subpackage contains definitions for the following domain adaptation algorithms:

-  Bridging Theory and Algorithm for Domain Adaptation (`DD`_)

-----------
Benchmarks
-----------

We provide benchmarks of different domain adaptation algorithms on `dSprites`_.

.. note::

    - ``Origin`` means the accuracy reported by the original paper.
    - ``Avg`` is the accuracy reported by DALIB.
    - ``Source Only`` refers to the model trained with data from the source domain.
    - ``Oracle`` refers to the model trained with data from the target domain.


.. note::

    Labels are all normalized to [0, 1] to eliminate the effects of diverse scale in regression values.

    We repeat experiments on DD for three times and report the average error of the ``final`` epoch.

.. _dSprites:

dSprites error on ResNet-18
---------------------------------
===========     ======  ======  ======  ======  ======  ======  ======
Methods         Avg     C → N   C → S   N → C   N → S   S → C   S → N
Source Only     0.157   0.232   0.271   0.081   0.220   0.038   0.092
DD              0.057   0.047	0.080	0.030	0.095	0.053	0.037
===========     ======  ======  ======  ======  ======  ======  ======

.. currentmodule:: dalib.adaptation

-----------
Algorithms
-----------

.. _DD:

DD/MDD
----------------------------
.. autoclass:: dalib.adaptation.mdd.RegressionMarginDisparityDiscrepancy

.. autoclass:: dalib.adaptation.mdd.ImageRegressor

