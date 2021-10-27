=======================================
Fine-tuning
=======================================

Regularization-based Fine-tuning
===========================================

.. _L2:

L2
------

.. autoclass:: talib.finetune.delta.L2Regularization


.. _L2SP:

L2-SP
------

.. autoclass:: talib.finetune.delta.SPRegularization


.. _DELTA:

DELTA: DEep Learning Transfer using Feature Map with Attention
-------------------------------------------------------------------------------------

.. autoclass:: talib.finetune.delta.BehavioralRegularization

.. autoclass:: talib.finetune.delta.AttentionBehavioralRegularization

.. autoclass:: talib.finetune.delta.IntermediateLayerGetter


.. _LWF:

LWF: Learning without Forgetting
------------------------------------------

.. autoclass:: talib.finetune.lwf.Classifier



.. _CoTuning:

Co-Tuning
------------------------------------------

.. autoclass:: talib.finetune.co_tuning.CoTuningLoss

.. autoclass:: talib.finetune.co_tuning.Relationship


.. _StochNorm:

StochNorm: Stochastic Normalization
------------------------------------------

.. autoclass:: talib.finetune.stochnorm.StochNorm1d

.. autoclass:: talib.finetune.stochnorm.StochNorm2d

.. autoclass:: talib.finetune.stochnorm.StochNorm3d

.. autofunction:: talib.finetune.stochnorm.convert_model


Adaptive Fine-tuning
===========================================

.. _BiTuning:

Bi-Tuning
------------------------------------------

.. autoclass:: talib.finetune.bi_tuning.BiTuning



Rejecting Untransferable Information
===========================================

.. _BSS:

BSS: Batch Spectral Shrinkage
------------------------------------------

.. autoclass:: talib.finetune.bss.BatchSpectralShrinkage

