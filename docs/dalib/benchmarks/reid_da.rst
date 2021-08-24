===========================================
Person Re-Identification Domain Adaptation
===========================================

We provide benchmarks of different domain adaptation algorithms. Currently three datasets are supported:
`Market1501 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410490>`_,
`DukeMTMC <https://arxiv.org/pdf/1609.01775v2.pdf>`_, `MSMT17 <https://arxiv.org/pdf/1711.08565.pdf>`_.
Those domain adaptation algorithms includes:

- :ref:`IBN`
- :ref:`MMT`

.. note::

    We adopt cross dataset setting (another one is cross camera setting). The model is first trained on source dataset,
    then we evaluate it on target dataset and report `mAP` (mean average precision) on target dataset.

.. note::
    For a fair comparison, our model is trained with standard cross entropy loss and triplet loss. We adopt modified
    resnet architecture from `Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised
    Domain Adaptation on Person Re-identification (ICLR 2020) <https://arxiv.org/pdf/2001.01526.pdf>`_.

.. note::
    As we are given unlabelled samples from target domain, we can utilize clustering algorithms to produce pseudo labels
    on target domain and then use them as supervision signals to perform self-training. This simple method turns out to
    be a strong baseline. We use ``Baseline_Cluster`` to represent this baseline in our results.

-----------------------------------
Cross dataset mAP on ResNet-50
-----------------------------------
================= ============= ============= ============= ============= =========== ===========
Methods            Market2Duke   Duke2Market   Market2MSMT   MSMT2Market   Duke2MSMT   MSMT2Duke
Baseline              32.4          31.4           8.2          36.7         11.0        43.1
IBN                   35.2          36.5
Baseline_Cluster
MMT                   63.0
================= ============= ============= ============= ============= =========== ===========
