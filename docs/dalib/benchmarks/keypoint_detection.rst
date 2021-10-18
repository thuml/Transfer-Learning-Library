==========================================
Keypoint Detection
==========================================

We provide benchmarks of different keypoint detection domain adaptation algorithms on  as follows.
Those domain adaptation algorithms includes:

-  :ref:`RegDA`

.. _RHD2H3D:

--------------------------------
RHD->H3D accuracy on ResNet-101
--------------------------------

===========     ======  ======  ======  =========   ======
Methods         MCP     PIP     DIP     Fingertip   Avg
Source Only     67.4	64.2	63.3	54.8	    61.8
RegDA           79.6	74.4	71.2	62.9	    72.5
Oracle          97.7	97.2	95.7	92.5	    95.8
===========     ======  ======  ======  =========   ======


.. _Surreal2Human36M:

-----------------------------------------
Surreal->Human3.6M accuracy on ResNet-101
-----------------------------------------

===========     ========    ======  ======  =====   =====   =====   =====
Methods         Shoulder    Elbow   Wrist   Hip     Knee    Ankle   Avg
Source Only     69.4        75.4    66.4    37.9    77.3    77.7    67.3
RegDA           73.3        86.4    72.8    54.8    82.0    84.4    75.6
Oracle          95.3        91.8    86.9    95.6    94.1    93.6    92.9
===========     ========    ======  ======  =====   =====   =====   =====


.. _Surreal2LSP:

-----------------------------------
Surreal->LSP accuracy on ResNet-101
-----------------------------------

===========     ========    ======  ======  =====   =====   =====   =====
Methods         Shoulder    Elbow   Wrist   Hip     Knee    Ankle   Avg
Source Only     51.5	    65.0    62.9    68.0    68.7    67.4    63.9
RegDA           62.7	    76.7    71.1    81.0    80.3    75.3    74.6
===========     ========    ======  ======  =====   =====   =====   =====


