=======================================
Self Training Methods
=======================================


.. _PseudoLabel:

Pseudo Label
-----------------------------

.. autoclass:: tllib.self_training.pseudo_label.ConfidenceBasedSelfTrainingLoss

.. _PiModel:

:math:`\Pi` Model
-----------------------------

.. autoclass:: tllib.self_training.pi_model.ConsistencyLoss


.. autoclass:: tllib.self_training.pi_model.L2ConsistencyLoss


.. _MeanTeacher:

Mean Teacher
-----------------------------

.. autoclass:: tllib.self_training.mean_teacher.EMATeacher


.. _SelfEnsemble:

Self Ensemble
-----------------------------

.. autoclass:: tllib.self_training.self_ensemble.ClassBalanceLoss


.. _UDA:

UDA
-----------------------------

.. autoclass:: tllib.self_training.uda.StrongWeakConsistencyLoss


.. _MCC:

MCC: Minimum Class Confusion
-----------------------------

.. autoclass:: tllib.self_training.mcc.MinimumClassConfusionLoss


.. _MMT:

MMT: Mutual Mean-Teaching
--------------------------
`Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised
Domain Adaptation on Person Re-identification (ICLR 2020) <https://arxiv.org/pdf/2001.01526.pdf>`_

State of the art unsupervised domain adaptation methods utilize clustering algorithms to generate pseudo labels on target
domain, which are noisy and thus harmful for training. Inspired by the teacher-student approaches, MMT framework
provides robust soft pseudo labels in an on-line peer-teaching manner.

We denote two networks as :math:`f_1,f_2`, their parameters as :math:`\theta_1,\theta_2`. The authors also
propose to use the temporally average model of each network :math:`\text{ensemble}(f_1),\text{ensemble}(f_2)` to generate more reliable
soft pseudo labels for supervising the other network. Specifically, the parameters of the temporally
average models of the two networks at current iteration :math:`T` are denoted as :math:`E^{(T)}[\theta_1]` and
:math:`E^{(T)}[\theta_2]` respectively, which can be calculated as

.. math::
    E^{(T)}[\theta_1] = \alpha E^{(T-1)}[\theta_1] + (1-\alpha)\theta_1
.. math::
    E^{(T)}[\theta_2] = \alpha E^{(T-1)}[\theta_2] + (1-\alpha)\theta_2

where :math:`E^{(T-1)}[\theta_1],E^{(T-1)}[\theta_2]` indicate the temporal average parameters of the two networks in
the previous iteration :math:`(T-1)`, the initial temporal average parameters are
:math:`E^{(0)}[\theta_1]=\theta_1,E^{(0)}[\theta_2]=\theta_2` and :math:`\alpha` is the momentum.

These two networks cooperate with each other in three ways:

- When running clustering algorithm, we average features produced by :math:`\text{ensemble}(f_1)` and
    :math:`\text{ensemble}(f_2)` instead of only considering one of them.
- A **soft triplet loss** is optimized between :math:`f_1` and :math:`\text{ensemble}(f_2)` and vice versa
    to force one network to learn from temporally average of another network.
- A **cross entropy loss** is optimized between :math:`f_1` and :math:`\text{ensemble}(f_2)` and vice versa
    to force one network to learn from temporally average of another network.

The above mentioned loss functions are listed below, more details can be found in training scripts.

.. autoclass:: tllib.vision.models.reid.loss.SoftTripletLoss

.. autoclass:: tllib.vision.models.reid.loss.CrossEntropyLoss


.. _SelfTuning:

Self Tuning
-----------------------------

.. autoclass:: tllib.self_training.self_tuning.Classifier

.. autoclass:: tllib.self_training.self_tuning.SelfTuning


.. _FlexMatch:

FlexMatch
-----------------------------

.. autoclass:: tllib.self_training.flexmatch.DynamicThresholdingModule
    :members:

.. _DST:

Debiased Self-Training
-----------------------------

.. autoclass:: tllib.self_training.dst.ImageClassifier

.. autoclass:: tllib.self_training.dst.WorstCaseEstimationLoss
