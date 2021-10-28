=======================================
Domain Adversarial Learning
=======================================

Domain Adversarial Training in Close-set DA
===========================================

.. _DANN:

DANN: Domain Adversarial Neural Network
----------------------------------------

.. autoclass:: dalib.adaptation.dann.DomainAdversarialLoss

.. _CDAN:

CDAN: Conditional Domain Adversarial Network
-----------------------------------------------

.. autoclass:: dalib.adaptation.cdan.ConditionalDomainAdversarialLoss


.. autoclass:: dalib.adaptation.cdan.RandomizedMultiLinearMap


.. autoclass:: dalib.adaptation.cdan.MultiLinearMap


.. _ADDA:

ADDA: Adversarial Discriminative Domain Adaptation
-----------------------------------------------------

.. autoclass:: dalib.adaptation.adda.DomainAdversarialLoss

.. note::
    ADDAgrl is also implemented and benchmarked. You can find code
    `here <https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/addagrl.py>`_.


.. _BSP:

BSP: Batch Spectral Penalization
-----------------------------------

.. autoclass:: dalib.adaptation.bsp.BatchSpectralPenalizationLoss


Domain Adversarial Training in Open-set DA
===========================================


.. _OSBP:

OSBP: Open Set Domain Adaptation by Backpropagation
----------------------------------------------------

.. autoclass:: dalib.adaptation.osbp.UnknownClassBinaryCrossEntropy




Domain Adversarial Training in Partial DA
===========================================

.. _PADA:

PADA: Partial Adversarial Domain Adaptation
---------------------------------------------

.. autoclass:: dalib.adaptation.pada.ClassWeightModule

.. autoclass:: dalib.adaptation.pada.AutomaticUpdateClassWeightModule
    :members:

.. autofunction::  dalib.adaptation.pada.collect_classification_results


.. _IWAN:

IWAN: Importance Weighted Adversarial Nets
---------------------------------------------

.. autoclass:: dalib.adaptation.iwan.ImportanceWeightModule
    :members:



Domain Adversarial Training in Segmentation
===========================================

.. _ADVENT:

ADVENT: Adversarial Entropy Minimization
------------------------------------------

.. autoclass:: dalib.adaptation.advent.Discriminator

.. autoclass:: dalib.adaptation.advent.DomainAdversarialEntropyLoss
    :members:
