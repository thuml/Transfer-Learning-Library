Image Classification (MoCo Pretrained on ImageNet)
========================================================

We provide benchmarks of different finetune algorithms on `CUB-200-2011`_, `StanfordCars`_,
`Aircraft`_.

Those domain adaptation algorithms includes:

-  :ref:`LWF`
-  :ref:`BSS`
-  :ref:`DELTA`
-  :ref:`CoTuning`
-  :ref:`StochNorm`
-  :ref:`BiTuning`


.. _CUB-200-2011:

------------------------------------
CUB-200-2011 accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        21.7	41.6	59.2	73.3
BSS
DELTA           21.5	45.4	61.5	70.3
StochNorm
Co-Tuning
Bi-Tuning       31.7	52.4	65.6	76.3
===========     ======  ======  ======  ======

.. _StanfordCars:

------------------------------------
Stanford Cars accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        39.7	68.4	81.9	90.6
BSS
DELTA
StochNorm
Co-Tuning
Bi-Tuning       45.3	72.1	84.0	90.6
===========     ======  ======  ======  ======

.. _Aircraft:

------------------------------------
FGVC Aircraft accuracy on ResNet-50
------------------------------------

===========     ======  ======  ======  ======
Methods         15%     30%     50%     100%
Baseline        41.7	67.2	76.9	86.6
BSS
DELTA
StochNorm
Co-Tuning
Bi-Tuning       45.8	69.4	80.3	87.7
===========     ======  ======  ======  ======
