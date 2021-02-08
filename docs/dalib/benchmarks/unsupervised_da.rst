===============================
Unsupervised Domain Adaptation
===============================

We provide benchmarks of different domain adaptation algorithms on `Office-31`_ , `Office-Home`_, `VisDA-2017`_  and  `DomainNet`_.
Those domain adaptation algorithms includes:

-  :ref:`DANN`
-  :ref:`DAN`
-  :ref:`JAN`
-  :ref:`CDAN`
-  :ref:`MCD`
-  :ref:`MDD`
-  :ref:`MCC`

.. note::

    - ``Origin`` means the accuracy reported by the original paper.
    - ``Avg`` is the accuracy reported by DALIB.
    - ``Source Only`` refers to the model trained with data from the source domain.
    - ``Oracle`` refers to the model trained with data from the target domain.

.. note::

    We found that the accuracies of adversarial methods (including DANN, CDAN, MCD and MDD) are not stable even after the random seed is fixed, thus
    we repeat running adversarial methods on *Office-31* and *VisDA-2017* for three times and report their average accuracy.


.. _Office-31:

--------------------------------
Office-31 accuracy on ResNet-50
--------------------------------

===========     ======  ======  ======  ======  ======  ======  ======  ======
Methods         Origin  Avg     A → W   D → W   W → D   A → D   D → A   W → A
Source Only     76.1	79.5	75.8	95.5	99.0	79.3	63.6	63.8
DANN            82.2	86.1	91.4	97.9	100.0	83.6	73.3	70.4
DAN             80.4	83.7	84.2	98.4	100.0	87.3	66.9	65.2
JAN             84.3	87.3	93.7	98.4	100.0	89.4	71.2	71.0
CDAN            87.7	87.7	93.8	98.5	100.0	89.9	73.4	70.4
MCD             /       85.4	90.4	98.5	100.0	87.3	68.3	67.6
MDD             88.9	89.6	95.6	98.6	100.0	94.4	76.6	72.2
MCC             89.4	89.6	94.1	98.4	99.8	95.6	75.5	74.2
===========     ======  ======  ======  ======  ======  ======  ======  ======


.. _Office-Home:

-----------------------------------
Office-Home accuracy on ResNet-50
-----------------------------------

=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin  Avg     Ar → Cl Ar → Pr Ar → Rw Cl → Ar Cl → Pr Cl → Rw Pr → Ar Pr → Cl Pr → Rw Rw → Ar Rw → Cl Rw → Pr
Source Only 46.1    58.4    41.1    65.9    73.7    53.1    60.1    63.3    52.2    36.7    71.8    64.8    42.6    75.2
DANN        57.6    65.2    53.8    62.6    74.0    55.8    67.3    67.3    55.8    55.1    77.9    71.1    60.7    81.1
DAN         56.3    61.4    45.6    67.7    73.9    57.7    63.8    66.0    54.9    40.0    74.5    66.2    49.1    77.9
JAN         58.3    65.9    50.8    71.9    76.5    60.6    68.3    68.7    60.5    49.6    76.9    71.0    55.9    80.5
CDAN        65.8    68.8    55.2    72.4    77.6    62.0    69.7    70.9    62.4    54.3    80.5    75.5    61.0    83.8
MCD         /       67.8    51.7    72.2    78.2    63.7    69.5    70.8    61.5    52.8    78.0    74.5    58.4    81.8
MDD         68.1    69.7    56.2    75.4    79.6    63.5    72.1    73.8    62.5    54.8    79.9    73.5    60.9    84.5
MCC         /       72.4    58.4    79.6    83.0    67.5    77.0    78.5    66.6    54.8    81.8    74.4    61.4    85.6
=========== ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

.. _VisDA-2017:

-----------------------------------
VisDA-2017 accuracy ResNet-101
-----------------------------------

.. note::
    - ``Origin`` means the accuracy reported by the original paper.
    - ``Mean`` refers to the accuracy average over ``classes``
    - ``Avg`` refers to accuracy average over ``samples``.

=========== ==========  ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin      Mean    plane   bcycl   bus     car     horse   knife   mcycl   person  plant   sktbrd  train   truck   Avg
Source Only 52.4        51.7    63.6    35.3    50.6    78.2    74.6    18.7    82.1    16.0    84.2    35.5    77.4    4.7     56.9
DANN        57.4        79.5	93.5	74.3	83.4	50.7	87.2	90.2	89.9	76.1	88.1	91.4	89.7	39.8	74.9
DAN         61.1        66.4	89.2	37.2	77.7	61.8	81.7	64.3	90.6	61.4	79.9	37.7	88.1	27.4	67.2
JAN         /           73.4	96.3	66.0	82.0	44.1	86.4	70.3	87.9	74.6	83.0	64.6	84.5	41.3	70.3
CDAN        /           80.1	94.0	69.2	78.9	57.0	89.8	94.9	91.9	80.3	86.8	84.9	85.0	48.5	76.5
MCD         71.9        77.7	87.8	75.7	84.2	78.1	91.6	95.3	88.1	78.3	83.4	64.5	84.8	20.9	76.7
MDD         /           82.0	88.3	62.8	85.2	69.9	91.9	95.1	94.4	81.2	93.8	89.8	84.1	47.9	79.8
MCC         78.8        83.6	95.3	85.8	77.1	68.0	93.9	92.9	84.5	79.5	93.6	93.7	85.3	53.8	80.4
=========== ==========  ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

.. _DomainNet:

-----------------------------------
DomainNet accuracy on ResNet-101
-----------------------------------

.. note::
    The column headings indicate the source domain, and the horizontal headings indicate the target domain.

Source Only
-----------

=========== ======  ======  ======  ======  ======  ======
Source Only clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    18.0    32.7    50.6    39.4    35.2
inf         35.7    N/A	    31.1    50.0    26.5    35.8
pnt         41.1    17.8    N/A     56.8    35.0    37.7
real        48.6    22.9    48.8    N/A	    36.1	39.1
skt         49.0    15.3    34.8    46.1    N/A     36.3
Avg         43.6    18.5    36.9    50.9    34.3    36.8
=========== ======  ======  ======  ======  ======  ======

DANN
-----------

=========== ======  ======  ======  ======  ======  ======
DANN        clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    19.7    35.4    53.9    44.2    38.3
inf         26.7    N/A     23.8    28.8    23.7    25.8
pnt         37.2    18.7    N/A     51.1    36.0    35.8
real        50.6    22.1    47.9    N/A     39.0    39.9
skt         54.0    19.7    42.7    52.8    N/A     42.3
Avg         42.1    20.1    37.5    46.7    35.7    36.4
=========== ======  ======  ======  ======  ======  ======

DAN
-----------

=========== ======  ======  ======  ======  ======  ======
DAN         clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    17.3    37.9    54.0    42.6    38.0
inf         34.9    N/A	    33.4    46.5    29.9    36.2
pnt         43.9    17.7    N/A     55.9    39.3    39.2
real        50.1    20.0    48.6    N/A	    38.4	39.3
skt         54.2    17.5    44.2    53.4    N/A     42.3
Avg         45.8    18.1    41.0    52.5    37.6    39.0
=========== ======  ======  ======  ======  ======  ======

CDAN
-----------

=========== ======  ======  ======  ======  ======  ======
CDAN        clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    20.8    40.0    56.1    45.5    40.6
inf         31.2    N/A	    30.0    41.4    24.7    31.8
pnt         44.6    20.5    N/A     57.0    39.9    40.5
real        55.3    24.1    52.6    N/A	    42.4	43.6
skt         56.7    21.3    46.2    55.0    N/A     44.8
Avg         47.0    21.7    42.2    52.4    38.1    40.3
=========== ======  ======  ======  ======  ======  ======

MDD
-----------

=========== ======  ======  ======  ======  ======  ======
MDD         clp	    inf	    pnt	    real    skt     Avg
clp         N/A	    21.2    42.9    59.5    47.5    42.8
inf         35.3    N/A	    34.0    49.6    29.4    37.1
pnt         48.6    19.7    N/A     59.4    42.6    42.6
real        58.3    24.9    53.7    N/A	    46.2	45.8
skt         58.7    20.7    46.5    57.7    N/A     45.9
Avg         50.2    21.6    44.3    56.6    41.4    42.8
=========== ======  ======  ======  ======  ======  ======

Oracle
-----------

=========== ======  ======  ======  ======  ======  ======
Oracle      clp	    inf	    pnt	    real    skt     Avg
/           78.2    40.7    71.6    83.8    70.6    69.0
=========== ======  ======  ======  ======  ======  ======
