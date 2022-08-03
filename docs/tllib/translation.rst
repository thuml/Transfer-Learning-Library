=======================================
Domain Translation
=======================================


.. _CycleGAN:

------------------------------------------------
CycleGAN: Cycle-Consistent Adversarial Networks
------------------------------------------------

Discriminator
--------------

.. autofunction:: tllib.translation.cyclegan.pixel

.. autofunction:: tllib.translation.cyclegan.patch

Generator
--------------

.. autofunction:: tllib.translation.cyclegan.resnet_9

.. autofunction:: tllib.translation.cyclegan.resnet_6

.. autofunction:: tllib.translation.cyclegan.unet_256

.. autofunction:: tllib.translation.cyclegan.unet_128


GAN Loss
--------------

.. autoclass:: tllib.translation.cyclegan.LeastSquaresGenerativeAdversarialLoss

.. autoclass:: tllib.translation.cyclegan.VanillaGenerativeAdversarialLoss

.. autoclass:: tllib.translation.cyclegan.WassersteinGenerativeAdversarialLoss

Translation
--------------

.. autoclass:: tllib.translation.cyclegan.Translation


Util
----------------

.. autoclass:: tllib.translation.cyclegan.util.ImagePool
    :members:

.. autofunction:: tllib.translation.cyclegan.util.set_requires_grad




.. _Cycada:

--------------------------------------------------------------
CyCADA: Cycle-Consistent Adversarial Domain Adaptation
--------------------------------------------------------------

.. autoclass:: tllib.translation.cycada.SemanticConsistency



.. _SPGAN:

-----------------------------------------------------------
SPGAN: Similarity Preserving Generative Adversarial Network
-----------------------------------------------------------
`Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification
<https://arxiv.org/pdf/1711.07027.pdf>`_. SPGAN is based on CycleGAN. An additional Siamese network is adopted to force
the generator to produce images different from identities in target dataset.

Siamese Network
-------------------

.. autoclass:: tllib.translation.spgan.siamese.SiameseNetwork

Contrastive Loss
-------------------

.. autoclass:: tllib.translation.spgan.loss.ContrastiveLoss


.. _FDA:

------------------------------------------------
FDA: Fourier Domain Adaptation
------------------------------------------------

.. autoclass:: tllib.translation.fourier_transform.FourierTransform

.. autofunction:: tllib.translation.fourier_transform.low_freq_mutate





