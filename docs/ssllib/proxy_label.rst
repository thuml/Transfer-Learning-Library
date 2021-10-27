=======================================
Proxy-Label Based Methods
=======================================

.. _PSEUDO:

Pseudo Label
------------------

Given model predictions :math:`y` on unlabeled samples, we can directly utilize them to generate
pseudo labels :math:`label=\mathop{\arg\max}\limits_{i}~y[i]`. Then we use these pseudo labels as supervision to train
our model. Details can be found at `projects/self_tuning/pseudo_label.py`.
