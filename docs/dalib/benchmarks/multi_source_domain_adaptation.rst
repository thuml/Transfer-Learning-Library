==========================================
Multi Source Domain Adaptation
==========================================

We provide benchmarks of different domain adaptation algorithms on `Office-Home`_ and  `DomainNet`_.
Those domain adaptation algorithms includes:

-  :ref:`MDD`

.. note::

    - ``Origin`` means the accuracy reported by the original paper.
    - ``Avg`` is the accuracy reported by Transfer-Learn.
    - ``Source Only`` refers to the model trained with data from the source domain.
    - ``Oracle`` refers to the model trained with data from the target domain.

.. _Office-Home:

Office-Home accuracy on ResNet-50
---------------------------------

===========     ======  ======  ======  ======  ======
Methods         Avg     :Ar     :Cl     :Pr     :Rw
Source Only     69.5    67.3    50.8    78.8    81.0
MDD             77.2    74.7    64.6    85.2    84.1
===========     ======  ======  ======  ======  ======

.. _DomainNet:

DomainNet accuracy on ResNet-50
-----------------------------------

=========== ======= ======= ======= ======= ======= ======= ======= =======
Methods     Origin  Avg     :c      :i      :p      :q      :r      :s
Source Only 32.9    47.0    64.9    25.2    54.4    16.9    68.2    52.3
MDD         /       48.8    68.7    29.7    58.2    9.7	    69.4    56.9
Oracle      63.0    69.1    78.2    40.7    71.6    69.7    83.8    70.6
=========== ======= ======= ======= ======= ======= ======= ======= =======
