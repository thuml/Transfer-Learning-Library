========================================================
Image Classification
========================================================

We provide benchmarks of different semi supervised learning algorithms on fine-grained classification datasets `CUB-200-2011`_, `StanfordCars`_
and `Aircraft`_.

Those semi supervised learning algorithms includes:

-  :ref:`PI_MODEL`
-  :ref:`MEAN_TEACHER`
-  :ref:`UDA`
-  :ref:`PSEUDO`
-  :ref:`FIXMATCH`
-  :ref:`SELF_TUNING`

We follow the common practice in the community as described in :ref:`SELF_TUNING`.
Hyper-parameters of each method are selected by the performance on target validation data.


.. _CUB-200-2011:

------------------------------------------------------------------------
CUB-200-2011 on ResNet-50 (Supervised Pre-trained)
------------------------------------------------------------------------

=============     ======  ======  ======  ======
Sample rate       15%     30%     50%     Avg
Baseline          46.8	  59.3	  69.8	  58.6
Pi Model          47.3	  59.2	  70.0	  58.8
Mean Teacher      63.8	  72.2	  78.4    71.5
UDA               46.8	  62.3	  72.2	  60.4
Pseudo Label      53.2	  64.9	  72.5	  63.5
FixMatch          49.8	  67.3	  75.9	  64.3
Self-Tuning       64.4    73.6	  78.8	  72.3
=============     ======  ======  ======  ======

.. _StanfordCars:

------------------------------------------------------------------------
Stanford Cars on ResNet-50 (Supervised Pre-trained)
------------------------------------------------------------------------

=============     ======  ======  ======  ======
Sample rate       15%     30%     50%     Avg
Baseline          38.3	  60.7	  72.6	  57.2
Pi Model          38.0	  59.7	  71.6	  56.4
Mean Teacher      70.5	  83.6	  89.1	  81.1
UDA               49.3	  69.6	  79.8	  66.2
Pseudo Label      45.6	  68.4	  78.6	  64.2
FixMatch          51.4	  76.1	  81.1	  69.5
Self-Tuning       74.7	  85.2    89.3    83.1
=============     ======  ======  ======  ======

.. _Aircraft:

------------------------------------------------------------------------
FGVC Aircraft on ResNet-50 (Supervised Pre-trained)
------------------------------------------------------------------------

=============     ======  ======  ======  ======
Sample rate       15%     30%     50%     Avg
Baseline          42.4	  59.0	  68.2	  56.5
Pi Model          43.0	  58.5	  69.0	  56.8
Mean Teacher      61.4	  76.0	  81.2	  72.9
UDA               50.5	  66.1	  73.0	  63.2
Pseudo Label      49.1	  68.4	  78.6	  65.4
FixMatch          50.0	  66.0	  71.9	  62.6
Self-Tuning
=============     ======  ======  ======  ======
