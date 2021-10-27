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
