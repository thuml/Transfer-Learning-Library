Classification (Pretrained on ImageNet)
=====================================================

We provide benchmarks of different finetune algorithms on `CUB-200-2011`_, `StanfordCars`_,
`Aircraft`_, `StanfordDogs`_, `Oxford-III-Pet`_ and `COCO-70`_ .

Those domain adaptation algorithms includes:

-  :ref:`BSS`
-  :ref:`DELTA`
-  :ref:`CoTuning`
-  :ref:`StochNorm`

.. note::

    We found that `StanfordDogs`_, `Oxford-III-Pet`_ and `COCO-70`_ have similar categories as ImageNet,
    thus most fine-tune algorithms take little effect on those datasets.
    Therefore we do not report the results.

.. _CUB-200-2011:

------------------------------------
CUB-200-2011 accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        46.7	59.6	70.3	79.0
BSS             49.3	62.8	72.4	79.9
DELTA           51.1	64.1	73.7	80.5
StochNorm       51.6	63.5	72.7	80.4
Co-Tuning       53.8	66.6	74.9	81.2
===========     ======  ======  ======  ======

.. _StanfordCars:

------------------------------------
Stanford Cars accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        40.0	63.8	76.8	87.7
BSS             43.5	67.5	78.3	88.0
DELTA           44.3	67.9	79.8	88.3
StochNorm       43.8	68.2	79.3	88.0
Co-Tuning       48.8	71.6	82.0	89.2
===========     ======  ======  ======  ======

.. _Aircraft:

------------------------------------
FGVC Aircraft accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        42.6	60.6	70.4	81.9
BSS             44.2	62.3	71.1	82.1
DELTA           46.4	63.2	71.7	82.3
StochNorm       46.7	63.2	71.7	81.9
Co-Tuning       45.7	62.3	72.5	83.0
===========     ======  ======  ======  ======

.. _StanfordDogs:

------------------------------------
Stanford Dogs accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        82.7	85.3	86.5	87.3
DELTA           84.5	86.4	87.3	88.2
===========     ======  ======  ======  ======

.. _Oxford-III-Pet:

------------------------------------
Oxford-III Pet accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        89.1	90.9	91.7	93.1
DELTA           90.6	91.9	92.8	93.7
===========     ======  ======  ======  ======

.. _COCO-70:

------------------------------------
COCO-70 accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        77.3	80.2	82.6	84.4
DELTA           79.2	81.7	83.5	84.6
===========     ======  ======  ======  ======