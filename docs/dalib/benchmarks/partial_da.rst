Partial Domain Adaptation
==========================================

We provide benchmarks of different domain adaptation algorithms on `Office-31`_ , `Office-Home`_, `VisDA-2017`_  and  `ImageNet-Caltech`_.
Those domain adaptation algorithms includes:

-  :ref:`DANN`
-  :ref:`PADA`

.. note::

    - ``Origin`` means the accuracy reported by the original paper.
    - ``Avg`` is the accuracy reported by DALIB.
    - ``Source Only`` refers to the model trained with data from the source domain.

.. note::

    We found that the accuracies of adversarial methods are not stable even after the random seed is fixed, thus
    we repeat running adversarial methods on *Office-31* and *VisDA-2017* for three times and report their average accuracy.

.. _Office-31:

Office-31 accuracy on ResNet-50
---------------------------------

===========     ======  ======  ======  ======  ======  ======  ======  ======
Methods         Origin  Avg     A → W   D → W   W → D   A → D   D → A   W → A
Source Only     75.6    90.1    78.3	98.3	99.4	87.3	88.5	88.8
DANN            43.4    82.4    60.0	94.9	98.1	71.3	84.9	85.0
PADA            92.7    93.8    86.4	100.0	100.0	87.3	93.8	95.4
===========     ======  ======  ======  ======  ======  ======  ======  ======

.. _Office-Home:

Office-Home accuracy on ResNet-50
-----------------------------------

=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin  Avg     Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 53.7    60.1    42.0    66.9    78.5    56.4    55.2    65.4    57.9    36.0    75.5    68.7    43.6    74.8
DANN        47.4    57.0    46.2    59.3    76.9    47.0    47.4    56.4    51.6    38.8    72.1    68.0    46.1    74.2
PADA        62.1    65.9    52.9    69.3    82.8    59.0    57.5    66.4    66.0    41.7    82.5    78.0    50.2    84.1
=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

.. _ImageNet-Caltech:

ImageNet-Caltech accuracy on ResNet-50
--------------------------------------

=========== ======= ======= ====    ====
Methods     Origin  Avg     I→C     C→I
Source Only 68.9    73.3    71.8	74.8
DANN        60.8    73.1    71.6	74.5
PADA        72.8    79.2    79.2	79.1
=========== ======= ======= ====    ====

.. _VisDA-2017:

VisDA-2017 accuracy on ResNet-50
-----------------------------------

Note that `Origin` means the accuracy reported by the original paper,
`Mean` refers to the accuracy average over classes, while `Avg` refers to accuracy average over samples.

=========== ==========  ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin      Mean    plane   bcycl   bus     car     horse   knife   Avg
Source Only 45.3        50.9	59.2	31.3	68.7	73.2	69.3	3.4	    60.0
DANN        51.0        55.9	88.4	34.1	72.1	50.7	61.9	27.8	57.1
PADA        53.5        60.5	89.4	35.1	72.5	69.2	86.7	10.1	66.8
=========== ==========  ======= ======= ======= ======= ======= ======= ======= =======
