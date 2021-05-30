
.. _CycleGAN:

------------------------------------------------
Cycle-Consistent Adversarial Networks (CycleGAN)
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

