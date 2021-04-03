*************
FAQ
*************

1. How to build the doc?
=========================

Sometimes, the online doc is not consistent with the latest code. In this case, you can build the doc by yourself.
First, you need to install sphinx, which is used to build doc.

.. code-block:: shell

    pip install -U sphinx

Second, you need to download a pytorch-style `html.zip <https://cloud.tsinghua.edu.cn/f/4d6b594de2694b399fb9/?dl=1>`_
into directory ``docs/build`` and unzip it. Some browsers may identify it as malicious file. You can safely ignore those warnings.

Then, in the directory ``docs`` run the following command

.. code-block:: shell

    make html

Also, warnings during ``make`` process doesn't matter and can be ignored.

Finally, you can open the docs in ``docs/build/html/index.html``

2. How to customize model?
===============================
A typical classifier has 3 components.

- backbone: usually ``ResNet`` network to extract feature maps.
- bottleneck: random initialized bottleneck layers between ``backbone`` and ``head`` to increase model capacity
- head: classifier head which outputs final predictions.

The way we construct an image classifier is as follows.

.. code-block:: python

    import common.modules.Classifier

    # define classifier
    classifier = Classifier(backbone=your_backbone, num_classes=num_classes, bottleneck=your_bottleneck,
        bottleneck_dim=your_bottleneck_dim, finetune=True)

Below are some examples on how to change these components.

2.1 Define a new backbone
'''''''''''''''''''''''''''
.. code-block:: python

    import torch.nn as nn


    class FeatureExtractor(nn.Module):

        def __init__(self):
            pass

        def forward(self, x):
            pass

        @property
        def out_features(self):
            pass

    your_backbone = FeatureExtractor()

Notice your ``FeatureExtractor`` should implement `out_features` method, which returns the dimension of output features.

2.2 Define a new bottleneck
'''''''''''''''''''''''''''
.. code-block:: python

    import torch.nn as nn

    your_bottleneck = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=(1, 1)), # for 2d images only
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU()
    )


2.3 Override get_parameters method
'''''''''''''''''''''''''''''''''''
In ``DA`` settings, we usually use smaller learning rates for ``backbone`` parameters, typically ``0.1x`` compared to other layers.
If you want to use other strategies such as different momentum, different weight_decay for different parts, you should inherit `Classifier` and
override ``get_parameters`` method.

Below we use same learning rate for ``backbone``, ``bottleneck`` and ``head``.
And momentum factor of ``0.9`` is used for ``bottleneck`` and ``head``.

.. code-block:: python

    import torch.nn as nn
    import common.modules.Classifier

    class ImageClassifier(Classifier):

        def __init__(self):
            pass

        def get_parameters(self):
            params = [
                {"params": self.backbone.parameters()},
                {"params": self.bottleneck.parameters(), "momentum": 0.9},
                {"params": self.head.parameters(), "momentum": 0.9},
            ]
            return params

3. How to customize datasets?
==================================

If you want to implement your own vision datasets, you can use ``torchvision.datasets.VisionDataset``
or ``common.vision.datasets.ImageList``.

Before using ``ImageList``, you need to prepare a txt file ``dog_cat.txt``.
In this file, each line should has the following format::

    path/to/dog/0.jpg 0
    path/to/cat/1.jpg 1

where the first part is a relative file path, and the second part is an integer label.

The way to construct an dog-cat dataset is

.. code-block:: python

    import torch.nn as nn
    import common.vision.datasets.ImageList

    dataset = ImageList(root="your_root", classes=("dog", "cat"), data_list_file="dog_cat.txt")

