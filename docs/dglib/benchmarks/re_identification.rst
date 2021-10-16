===============================
Re-Identification
===============================

We provide benchmarks of different domain generalization algorithms. Currently three datasets are supported:
`Market1501 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410490>`_,
`DukeMTMC <https://arxiv.org/pdf/1609.01775v2.pdf>`_, `MSMT17 <https://arxiv.org/pdf/1711.08565.pdf>`_.
Those domain generalization algorithms includes:

- :ref:`IBN`
- :ref:`MIXSTYLE`

.. note::

    We adopt cross dataset setting (another one is cross camera setting). The model is first trained on source dataset,
    then we evaluate it on target dataset and report `mAP` (mean average precision) on target dataset.

.. note::
    For a fair comparison, our model is trained with standard cross entropy loss and triplet loss. We adopt modified
    resnet architecture from `Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised
    Domain Adaptation on Person Re-identification (ICLR 2020) <https://arxiv.org/pdf/2001.01526.pdf>`_.

-----------------------------------
Cross dataset mAP on ResNet-50
-----------------------------------
======== ======= ============= ============= ============= ============= =========== ===========
Methods    Avg    Market2Duke   Duke2Market   Market2MSMT   MSMT2Market   Duke2MSMT   MSMT2Duke
Baseline   23.5     25.6          29.6           6.3          31.7          10.1       37.8
IBN        27.0     31.5          33.3           10.4         33.6          13.7       40.0
MixStyle   25.5     27.2          31.6           8.2          33.9          12.4       39.9
======== ======= ============= ============= ============= ============= =========== ===========
