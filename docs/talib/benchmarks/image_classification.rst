========================================================
Image Classification
========================================================

We provide benchmarks of different task adaptation algorithms on fine-grained classification datasets `CUB-200-2011`_, `StanfordCars`_,
`Aircraft`_, and specialized datasets `Resisc45`_,  `PatchCamelyon`_.

Those task adaptation algorithms includes:

-  :ref:`LWF`
-  :ref:`L2SP`
-  :ref:`BSS`
-  :ref:`DELTA`
-  :ref:`CoTuning`
-  :ref:`StochNorm`
-  :ref:`BiTuning`

We follow the common practice in the community as described in :ref:`BSS`.
Training iterations and data augmentations are kept the same for different task-adaptation
methods for a fair comparison.
Hyper-parameters of each method are selected by the performance on target validation data.


.. _CUB-200-2011:

------------------------------------------------------------------------
CUB-200-2011 on ResNet-50 (Supervised Pre-trained)
------------------------------------------------------------------------

===========     ======  ======  ======  ======  ======
Sample rate     15%     30%     50%     100%    Avg
Baseline        51.2	64.6	74.6	81.8    68.1
LWF             56.7	66.8	73.4	81.5	69.6
BSS             53.4	66.7	76.0	82.0	69.5
DELTA           54.8	67.3	76.3	82.3	70.2
StochNorm       54.8	66.8	75.8	82.2	69.9
Co-Tuning       57.6	70.1	77.3	82.5	71.9
Bi-Tuning       55.8	69.3	77.2	83.1	71.4
===========     ======  ======  ======  ======  ======

.. _StanfordCars:

------------------------------------------------------------------------
Stanford Cars on ResNet-50 (Supervised Pre-trained)
------------------------------------------------------------------------

===========     ======  ======  ======  ======  ======
Sample rate     15%     30%     50%     100%    Avg
Baseline        41.1	65.9	78.4	87.8	68.3
LWF             44.9	67.0	77.6	87.5	69.3
BSS             43.3	67.6	79.6	88.0	69.6
DELTA           45.0	68.4	79.6	88.4	70.4
StochNorm       44.4	68.1	79.3	87.9	69.9
Co-Tuning       49.0	70.6	81.9	89.1	72.7
Bi-Tuning       48.3	72.8	83.3	90.2	73.7
===========     ======  ======  ======  ======  ======

.. _Aircraft:

------------------------------------------------------------------------
FGVC Aircraft on ResNet-50 (Supervised Pre-trained)
------------------------------------------------------------------------

===========     ======  ======  ======  ======  ======
Sample rate     15%     30%     50%     100%    Avg
Baseline        41.6	57.8	68.7	80.2	62.1
LWF             44.1	60.6	68.7	82.4	64.0
BSS             43.6	59.5	69.6	81.2	63.5
DELTA           44.4	61.9	71.4	82.7	65.1
StochNorm       44.3	60.6	70.1	81.5	64.1
Co-Tuning       45.9	61.2	71.3	82.2	65.2
Bi-Tuning       47.2	64.3	73.7	84.3	67.4
===========     ======  ======  ======  ======  ======

Resisc45 and PatchCamelyon have very different data distributions from ImageNet,
thus some regularization-based fine-tuning methods lead to negative transfer on these two benchmarks.

.. _Resisc45:

------------------------------------------------------------------------
Resisc45 on ResNet-50 (Supervised Pre-trained)
------------------------------------------------------------------------

=================   ======  ======  ======  ======  ======
#samples/#classes   10      20      40      80      Avg
Baseline            74.0    81.9    86.5    90.1    83.1
LWF                 74.2    81.7    86.1    89.9    83.0
BSS                 73.5    80.9    85.7    90.0    82.5
DELTA               73.9    81.7    86.0    90.1    82.9
StochNorm           73.9    82.1    87.0    90.2    83.3
Co-Tuning           75.0    82.7    87.3    91.4    84.1
Bi-Tuning           75.3    83.2    87.4    91.4    84.3
=================   ======  ======  ======  ======  ======


.. _PatchCamelyon:

------------------------------------------------------------------------
PatchCamelyon on ResNet-50 (Supervised Pre-trained)
------------------------------------------------------------------------

=================   ======  ======  ======  ======  ======
#samples/#classes   40      80      160     320     Avg
Baseline            75.5    75.9    80.4    81.2    78.3
LWF                 75.3    77.7    80.6    82.5    79.0
BSS                 78.0    78.2    80.4    80.4    79.3
DELTA               74.1    76.4    80.0    81.9    78.1
StochNorm           75.9    77.1    78.2    81.3    78.1
Co-Tuning           75.1    76.2    80.7    81.8    78.5
Bi-Tuning           75.1    77.6    80.6    81.4    78.7
=================   ======  ======  ======  ======  ======

We further evaluate task adaptation algorithms when the downstream tasks are different
from the pre-training tasks. The pre-training task is MoCo unsupervised pre-training, and
the downstream tasks are still fine-grained classification.
In this scenario, some regularization-based fine-tuning methods also lead to negative transfer.

.. _CUB-200-2011_MoCo:

------------------------------------------------------------------------
CUB-200-2011 on ResNet-50 (MoCo Pre-trained)
------------------------------------------------------------------------

===========     ======  ======  ======  ======  ======
Sample rate     15%     30%     50%     100%    Avg
Baseline        28.0	48.2	62.7	75.6	53.6
LWF             28.8	50.1	62.8	76.2	54.5
BSS             30.9	50.3	63.7	75.8	55.2
DELTA           27.9	51.4	65.9	74.6	55.0
StochNorm       20.8	44.9	60.1	72.8	49.7
Co-Tuning       29.1	50.1	63.8	75.9	54.7
Bi-Tuning       32.4	51.8	65.7	76.1	56.5
===========     ======  ======  ======  ======  ======

.. _StanfordCars_MoCo:

------------------------------------------------------------------------
Stanford Cars on ResNet-50 (MoCo Pre-trained)
------------------------------------------------------------------------

===========     ======  ======  ======  ======  ======
Sample rate     15%     30%     50%     100%    Avg
Baseline        42.5	71.2	83.0	90.1	71.7
LWF             44.2	71.7	82.9	90.5	72.3
BSS             45.0	71.5	83.8	90.1	72.6
DELTA           45.9	72.9	82.5	88.9	72.6
StochNorm       40.3	66.2	78.0	86.2	67.7
Co-Tuning       44.2	72.6	83.3	90.3	72.6
Bi-Tuning       45.6	72.8	83.2	90.8	73.1
===========     ======  ======  ======  ======  ======

.. _Aircraft_MoCo:

------------------------------------------------------------------------
FGVC Aircraft on ResNet-50 (MoCo Pre-trained)
------------------------------------------------------------------------

===========     ======  ======  ======  ======  ======
Sample rate     15%     30%     50%     100%    Avg
Baseline        45.8	67.6	78.8	88.0	70.1
LWF             48.5	68.5	78.0	87.9	70.7
BSS             47.7	69.1	79.2	88.0	71.0
DELTA           \-      \-      \-      \-      \-
StochNorm       45.4	68.8	76.7	86.1	69.3
Co-Tuning       48.2	68.5	78.7	87.3	70.7
Bi-Tuning       46.4	69.6	79.4	87.9	70.8
===========     ======  ======  ======  ======  ======

.. note::
    \- indicates that the training cannot converge.
