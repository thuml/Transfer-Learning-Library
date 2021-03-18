Fine-grained Classification
==========================================

We provide benchmarks of different finetune algorithms on `CUB-200-2011`_, `StanfordCars`_,
`Aircraft`_, `StanfordDogs`_, `Oxford-III-Pet`_ and `COCO-70`_ .
Those domain adaptation algorithms includes:

-  :ref:`BSS`
-  :ref:`DELTA`
-  :ref:`CoTuning`
-  :ref:`StochNorm`

.. _CUB-200-2011:

------------------------------------
CUB-200-2011 accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        46.7	59.6	70.3	79.0
BSS             49.3	62.8	72.4	79.9
DELTA           50.7	64.5	74.1	80.3
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
DELTA           45.1	69.5	80.3	89.1
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
DELTA           46.8	64.7	72.5	83.3
StochNorm       46.7	63.2	71.7	81.9
Co-Tuning       45.7	62.3	72.5	83.0
===========     ======  ======  ======  ======

.. _StanfordDogs:

------------------------------------
Stanford Dogs accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
BSS
DELTA
StochNorm
Co-Tuning
===========     ======  ======  ======  ======

.. _Oxford-III-Pet:

------------------------------------
Oxford-III Pet accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
BSS
DELTA
StochNorm
Co-Tuning
===========     ======  ======  ======  ======

.. _COCO-70:

------------------------------------
COCO-70 accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
BSS
DELTA
StochNorm
Co-Tuning
===========     ======  ======  ======  ======
