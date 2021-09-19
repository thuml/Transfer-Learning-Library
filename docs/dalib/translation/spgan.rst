
.. _SPGAN:

-----------------------------------------------------------
Similarity Preserving Generative Adversarial Network (SPGAN)
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
