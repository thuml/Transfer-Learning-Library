Segmentation Domain Adaptation
==========================================

The segmentation adaptation subpackage contains definitions for the following domain adaptation algorithms:

-  `ADVENT`_
-  `FDA`_

-----------
Benchmarks
-----------

We provide benchmarks of different segmentation domain adaptation algorithms on *GTA5->Cityscapes* and *Synthia->Cityscapes* as follows.
Note that `Origin` means the accuracy reported by the original paper, while `DALIB` is the accuracy reported by DALIB.


.. _ADVENT: https://arxiv.org/abs/1811.12833
.. _FDA: https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf


-----------
Algorithms
-----------


ADVENT
----------------------------

.. autoclass:: dalib.adaptation.segmentation.advent.Discriminator

.. autoclass:: dalib.adaptation.segmentation.advent.DomainAdversarialEntropyLoss
    :members:


FDA
----------------------------

.. autoclass:: dalib.translation.fourier_transform.FourierTransform

.. autofunction:: dalib.translation.fourier_transform.low_freq_mutate


.. autofunction:: dalib.adaptation.segmentation.fda.robust_entropy



