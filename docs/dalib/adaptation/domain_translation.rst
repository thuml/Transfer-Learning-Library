=======================================
Domain Translation
=======================================


.. _CycleGAN:

------------------------------------------------
CycleGAN: Cycle-Consistent Adversarial Networks
------------------------------------------------

Discriminator
--------------

.. autofunction:: dalib.translation.cyclegan.pixel

.. autofunction:: dalib.translation.cyclegan.patch

Generator
--------------

.. autofunction:: dalib.translation.cyclegan.resnet_9

.. autofunction:: dalib.translation.cyclegan.resnet_6

.. autofunction:: dalib.translation.cyclegan.unet_256

.. autofunction:: dalib.translation.cyclegan.unet_128


GAN Loss
--------------

.. autoclass:: dalib.translation.cyclegan.LeastSquaresGenerativeAdversarialLoss

.. autoclass:: dalib.translation.cyclegan.VanillaGenerativeAdversarialLoss

.. autoclass:: dalib.translation.cyclegan.WassersteinGenerativeAdversarialLoss

Translation
--------------

.. autoclass:: dalib.translation.cyclegan.Translation


Util
----------------

.. autoclass:: dalib.translation.cyclegan.util.ImagePool
    :members:

.. autofunction:: dalib.translation.cyclegan.util.set_requires_grad




.. _Cycada:

--------------------------------------------------------------
CyCADA: Cycle-Consistent Adversarial Domain Adaptation
--------------------------------------------------------------

.. autoclass:: dalib.translation.cycada.SemanticConsistency



.. _SPGAN:

-----------------------------------------------------------
SPGAN: Similarity Preserving Generative Adversarial Network
-----------------------------------------------------------
`Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification
<https://arxiv.org/pdf/1711.07027.pdf>`_. SPGAN is based on CycleGAN. An additional Siamese network is adopted to force
the generator to produce images different from identities in target dataset.

Siamese Network
-------------------

.. autoclass:: dalib.translation.spgan.siamese.SiameseNetwork

Contrastive Loss
-------------------

.. autoclass:: dalib.translation.spgan.loss.ContrastiveLoss


.. _FDA:

------------------------------------------------
FDA: Fourier Domain Adaptation
------------------------------------------------

.. autoclass:: dalib.translation.fourier_transform.FourierTransform

.. autofunction:: dalib.translation.fourier_transform.low_freq_mutate

.. autofunction:: dalib.adaptation.fda.robust_entropy




