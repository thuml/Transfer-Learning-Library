*************
Visualization
*************

Visualization is a powerful tool, especially in `DA` settings. Many algorithms aim to align feature representations
between ``source`` and ``target`` domain. ``DANN`` globally aligns feature representations. As a result, you can
find there may exist mis-alignment between classes through visualization. Another example is `AFN <https://arxiv.org/pdf/1811.07456v2.pdf>`_.
Authors of ``AFN`` find features with larger norms tend to be more transferable, then they develop their simple but effective algorithm.

Next we show you how to visualize results with our ``dalib`` framework.

Image Classification
=====================
Again, we use `DANN`, our old friend. After training, in directory ``examples-da/unsupervised``, run the following command

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31
        -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --phase analysis

It may take a while, then in directory ``examples-da/unsupervised/dann/Office31_A2W/visualize``, you can find
``TSNE.png``.

.. image:: /docs/_static/images/dann_A2W.png

