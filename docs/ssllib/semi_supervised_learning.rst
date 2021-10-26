=======================================
Semi Supervised Learning
=======================================

Consistency Regularization
=======================================

.. _PI_MODEL:

Pi Model
------------------

.. autofunction:: ssllib.pi_model.sigmoid_rampup

.. autofunction:: ssllib.pi_model.softmax_mse_loss

.. autofunction:: ssllib.pi_model.symmetric_mse_loss

.. autoclass:: ssllib.pi_model.SoftmaxMSELoss

.. autoclass:: ssllib.pi_model.SoftmaxKLLoss


.. _MEAN_TEACHER:

Mean Teacher
------------------

.. autofunction:: ssllib.mean_teacher.update_ema_variables

.. autoclass:: ssllib.mean_teacher.SymmetricMSELoss

.. autoclass:: ssllib.mean_teacher.MeanTeacher


.. _UDA:

Unsupervised Data Augmentation (UDA)
------------------------------------

.. autoclass:: ssllib.rand_augment.RandAugment

.. autoclass:: ssllib.uda.SupervisedUDALoss

.. autoclass:: ssllib.uda.UnsupervisedUDALoss


Pseudo Labels
=======================================

.. _PSEUDO:

Pseudo Label
------------------

Given model predictions :math:`y` on unlabeled samples, we can directly utilize them to generate
pseudo labels :math:`label=\mathop{\arg\max}\limits_{i}~y[i]`. Then we use these pseudo labels as supervision to train
our model. Details can be found at `projects/self_tuning/pseudo_label.py`.


Holistic Methods
=======================================

.. _FIXMATCH:

FixMatch
------------------

.. autoclass:: ssllib.fix_match.FixMatchConsistencyLoss


Contrastive Learning
=======================================

.. _SELF_TUNING:

Self-Tuning
------------------

.. autoclass:: ssllib.self_tuning.Classifier

.. autoclass:: ssllib.self_tuning.SelfTuning
