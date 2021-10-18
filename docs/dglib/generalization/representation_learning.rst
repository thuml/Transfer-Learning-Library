=======================================
Representation Learning
=======================================

.. _CORAL:

Correlation Alignment for Deep Domain Adaptation (Deep CORAL)
--------------------------------------------------------------

.. autoclass:: dglib.generalization.coral.CorrelationAlignmentLoss

.. note::
    Under domain generalization setting, we can't access to samples from target domain during training. Therefore,
    two different domains from source domains are treated as source domain and target domain respectively to calculate
    `Correlation Alignment Loss`.
