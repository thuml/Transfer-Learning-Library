===========================================
Re-Identification
===========================================

We provide benchmarks of different domain adaptation algorithms. Currently three datasets are supported:
`Market1501 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410490>`_,
`DukeMTMC <https://arxiv.org/pdf/1609.01775v2.pdf>`_, `MSMT17 <https://arxiv.org/pdf/1711.08565.pdf>`_.
Those domain adaptation algorithms includes:

- :ref:`IBN`
- :ref:`SPGAN`
- :ref:`MMT`


We adopt cross dataset setting (another one is cross camera setting). The model is first trained on source dataset, then we evaluate it on target dataset and report `mAP` (mean average precision) on target dataset.

For a fair comparison, our model is trained with standard cross entropy loss and triplet loss. We adopt modified resnet architecture from `Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification (ICLR 2020) <https://arxiv.org/pdf/2001.01526.pdf>`_.

As we are given unlabeled samples from target domain, we can utilize clustering algorithms to produce pseudo labels on target domain and then use them as supervision signals to perform self-training. This simple method turns out to be a strong baseline. We use ``Baseline_Cluster`` to represent this baseline in our results.

.. note::

    - ``Avg`` means the average mAP across these tasks reported by Transfer-Learn.

-----------------------------------
Cross dataset mAP on ResNet-50
-----------------------------------
========================= ======= ============= ============= ============= ============= =========== ===========
Methods                     Avg    Market2Duke   Duke2Market   Market2MSMT   MSMT2Market   Duke2MSMT   MSMT2Duke
Baseline                   27.1       32.4          31.4           8.2          36.7         11.0        43.1
IBN                        30.0       35.2          36.5           11.3         38.7         14.1        44.3
SPGAN                      30.7       34.4          35.4           14.1         40.2         16.1        43.8
Baseline_Cluster(kmeans)   45.1       52.8          59.5           19.0         62.6         20.3        56.2
Baseline_Cluster(dbscan)   54.9       62.5          73.5           25.2         77.9         25.3        65.0
MMT(kmeans)                55.4       63.7          72.5           26.2         75.8         28.0        66.1
MMT(dbscan)                60.0       68.2          80.0           28.2         82.5         31.2        70.0
========================= ======= ============= ============= ============= ============= =========== ===========
