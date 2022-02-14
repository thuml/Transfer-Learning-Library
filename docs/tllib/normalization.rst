=====================
Normalization
=====================



.. _AFN:

AFN: Adaptive Feature Norm
-----------------------------

.. autoclass:: tllib.normalization.afn.AdaptiveFeatureNorm

.. autoclass:: tllib.normalization.afn.Block

.. autoclass:: tllib.normalization.afn.ImageClassifier


StochNorm: Stochastic Normalization
------------------------------------------

.. autoclass:: tllib.normalization.stochnorm.StochNorm1d

.. autoclass:: tllib.normalization.stochnorm.StochNorm2d

.. autoclass:: tllib.normalization.stochnorm.StochNorm3d

.. autofunction:: tllib.normalization.stochnorm.convert_model


.. _IBN:

IBN-Net: Instance-Batch Normalization Network
------------------------------------------------

.. autoclass:: tllib.normalization.ibn.InstanceBatchNorm2d

.. autoclass:: tllib.normalization.ibn.IBNNet
    :members:

.. automodule:: tllib.normalization.ibn
   :members:


.. _MIXSTYLE:

MixStyle: Domain Generalization with MixStyle
-------------------------------------------------

.. autoclass:: tllib.normalization.mixstyle.MixStyle

.. note::
    MixStyle is only activated during `training` stage, with some probability :math:`p`.

.. automodule:: tllib.normalization.mixstyle.resnet
    :members:
