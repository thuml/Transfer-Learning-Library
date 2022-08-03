==========================================
Domain Adversarial Training
==========================================


.. _DANN:

DANN: Domain Adversarial Neural Network
----------------------------------------

.. autoclass:: tllib.alignment.dann.DomainAdversarialLoss

.. _CDAN:

CDAN: Conditional Domain Adversarial Network
-----------------------------------------------

.. autoclass:: tllib.alignment.cdan.ConditionalDomainAdversarialLoss


.. autoclass:: tllib.alignment.cdan.RandomizedMultiLinearMap


.. autoclass:: tllib.alignment.cdan.MultiLinearMap


.. _ADDA:

ADDA: Adversarial Discriminative Domain Adaptation
-----------------------------------------------------

.. autoclass:: tllib.alignment.adda.DomainAdversarialLoss

.. note::
    ADDAgrl is also implemented and benchmarked. You can find code
    `here <https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/addagrl.py>`_.


.. _BSP:

BSP: Batch Spectral Penalization
-----------------------------------

.. autoclass:: tllib.alignment.bsp.BatchSpectralPenalizationLoss


.. _OSBP:

OSBP: Open Set Domain Adaptation by Backpropagation
----------------------------------------------------

.. autoclass:: tllib.alignment.osbp.UnknownClassBinaryCrossEntropy


.. _ADVENT:

ADVENT: Adversarial Entropy Minimization for Semantic Segmentation
------------------------------------------------------------------

.. autoclass:: tllib.alignment.advent.Discriminator

.. autoclass:: tllib.alignment.advent.DomainAdversarialEntropyLoss
    :members:


.. _DADAPT:

D-adapt: Decoupled Adaptation for Cross-Domain Object Detection
----------------------------------------------------------------
`Origin Paper <https://openreview.net/pdf?id=VNqaB1g9393>`_.

.. autoclass:: tllib.alignment.d_adapt.proposal.Proposal

.. autoclass:: tllib.alignment.d_adapt.proposal.PersistentProposalList

.. autoclass:: tllib.alignment.d_adapt.proposal.ProposalDataset

.. autoclass:: tllib.alignment.d_adapt.modeling.meta_arch.DecoupledGeneralizedRCNN

.. autoclass:: tllib.alignment.d_adapt.modeling.meta_arch.DecoupledRetinaNet

