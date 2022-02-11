==========================================
Hypothesis Adversarial Learning
==========================================



.. _MCD:

MCD: Maximum Classifier Discrepancy
--------------------------------------------

.. autofunction:: tllib.alignment.mcd.classifier_discrepancy

.. autofunction:: tllib.alignment.mcd.entropy

.. autoclass:: tllib.alignment.mcd.ImageClassifierHead


.. _MDD:


MDD: Margin Disparity Discrepancy
--------------------------------------------


.. autoclass:: tllib.alignment.mdd.MarginDisparityDiscrepancy


**MDD for Classification**


.. autoclass:: tllib.alignment.mdd.ClassificationMarginDisparityDiscrepancy


.. autoclass:: tllib.alignment.mdd.ImageClassifier
    :members:

.. autofunction:: tllib.alignment.mdd.shift_log


**MDD for Regression**

.. autoclass:: tllib.alignment.mdd.RegressionMarginDisparityDiscrepancy

.. autoclass:: tllib.alignment.mdd.ImageRegressor


.. _RegDA:

RegDA: Regressive Domain Adaptation
--------------------------------------------

.. autoclass:: tllib.alignment.regda.PseudoLabelGenerator2d

.. autoclass:: tllib.alignment.regda.RegressionDisparity

.. autoclass:: tllib.alignment.regda.PoseResNet2d
