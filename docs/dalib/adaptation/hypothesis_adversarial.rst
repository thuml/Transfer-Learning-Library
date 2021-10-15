=======================================
Hypothesis Adversarial Learning
=======================================

.. _MCD:

MCD: Maximum Classifier Discrepancy
=============================================

.. autofunction:: dalib.adaptation.mcd.classifier_discrepancy

.. autofunction:: dalib.adaptation.mcd.entropy

.. autoclass:: dalib.adaptation.mcd.ImageClassifierHead


.. _MDD:

MDD: Margin Disparity Discrepancy
=============================================


.. autoclass:: dalib.adaptation.mdd.MarginDisparityDiscrepancy


MDD for Classification
----------------------

.. autoclass:: dalib.adaptation.mdd.ClassificationMarginDisparityDiscrepancy


.. autoclass:: dalib.adaptation.mdd.ImageClassifier
    :members:

.. autofunction:: dalib.adaptation.mdd.shift_log


MDD for Regression
------------------

.. autoclass:: dalib.adaptation.mdd.RegressionMarginDisparityDiscrepancy

.. autoclass:: dalib.adaptation.mdd.ImageRegressor


.. _RegDA:

RegDA: Regressive Domain Adaptation
=============================================

.. autoclass:: dalib.adaptation.regda.PseudoLabelGenerator2d

.. autoclass:: dalib.adaptation.regda.RegressionDisparity

.. autoclass:: dalib.adaptation.regda.PoseResNet2d

