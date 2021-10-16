==========================================
Openset Domain Adaptation
==========================================

We provide benchmarks of different domain adaptation algorithms on `Office-31`_ , `Office-Home`_ and `VisDA-2017`_ as follows.
Those domain adaptation algorithms includes:

-  :ref:`DANN`
-  :ref:`OSBP`


.. note::
    - ``Source Only`` refers to the model trained with data from the source domain.
    - ``OS`` means normalized accuracy for all classes including the unknown as one class.
    - ``OS*`` means normalized accuracy only on known classes.
    - ``UNK`` is the accuracy of unknown samples.

    In ``OS``, the accuracy of each common class has the same contribution
    as the whole ``unknown`` class. Thus we report ``HOS`` used in `ROS (ECCV 2020)`_
    to better measure the abilities of different open set domain adaptation algorithms.

    .. math::
        \textit{HOS} = 2 \cdot \dfrac{ \textit{OS*} \cdot \textit{UNK} }{ \textit{OS*} + \textit{UNK} }

    The new evaluation metric is high only when both the ``OS*`` and ``UNK`` are high.

.. note::
    We report the best ``HOS`` in all epochs.

    DANN (baseline model) will degrade performance as training progresses, thus the
    final ``HOS`` will be much lower than reported.
    In contrast, OSBP will improve performance stably.

.. _Office-31:

Office-31 H-Score on ResNet-50
---------------------------------

.. note::
    We conduct `21` class classification experiments in this setting (follows :ref:`OSBP`).

===========     =====   ======  ======  ======  ======  ======  ======
Methods         Avg     A → W   D → W   W → D   A → D   D → A   W → A
Source Only     75.9    67.7    85.7    91.4    72.1    68.4    67.8
DANN            80.4    81.4    89.1    92.0    82.5    66.7    70.4
OSBP            87.8    90.7    96.4    97.5    88.7    77.0    76.7
===========     =====   ======  ======  ======  ======  ======  ======

.. _Office-Home:

.. note::

    - ``Origin`` means the accuracy reported by the related paper.
    - ``Avg`` is the accuracy reported by DALIB.

Office-Home HOS on ResNet-50
-----------------------------------

=========== ======  ======  ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin  Avg     Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only /       59.8    55.2    65.2    71.4    52.8    59.6    65.2    55.8    44.8    68.0    63.8    49.4    68.0
DANN        /       64.8    55.2    65.2    71.4    52.8    59.6    65.2    55.8    44.8    68.0    63.8    49.4    68.0
OSBP        64.7    68.6    62.0    70.8    76.5    66.4    68.8    73.8    65.8    57.1    75.4    70.6    60.6    75.9
=========== ======  ======  ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

.. _VisDA-2017:

VisDA-2017 performance on ResNet-50
-----------------------------------

=========== ========    ======  =====   ====    ======= ======= ======= ======= ======= =======
Methods     HOS         OS      OS*     UNK     bcycl   bus     car     mcycl   train   truck
Source Only 42.6        37.6    34.7    55.1    42.6    6.4     30.5    67.1    84.0    0.2
DANN        57.8        50.4    45.6    78.9    20.1	71.4	29.5	74.4	67.8	10.4
OSBP        75.4        67.3    62.9    94.3    63.7	75.9	49.6	74.4	86.2	27.3
=========== ========    ======  =====   ====    ======= ======= ======= ======= ======= =======


.. _ROS (ECCV 2020): http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610409.pdf
