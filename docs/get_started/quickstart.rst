************
QuickStart
************

This section will walk you through pipelines to reproduce the benchmark results.

Before going deeper, please **make sure** you have installed all the dependency.

Below, `DANN <https://arxiv.org/abs/1505.07818>`_ is used as an example.

Step1: Find it
===================
Our code for domain adaptation is in the directory ``examples-da``.

DANN aims to address close set domain adaptation tasks for image classification. You can find it in
``examples-da/unsupervised``. This directory contains implements for other algorithms like ``CDAN``, ``MDD`` as well.
For now, we only care about files about ``DANN``. Two files here are: ``dann.py`` and ``dann.sh``.

Step2: Run it
===================
``dann.py`` is an executable python file. And you can find commands to execute it in ``dann.sh``.

For instance, We use the following command.

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31
        -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W

You may get confused about these arguments. And some of them are easy to guess. ``-s`` specifies source domain.
``-t`` specifies target domain. ``--log`` specifies where to store results. Don't worry, we will go into that later in
Step 4.

In directory ``examples-da/unsupervised``, run above command. You will find it starts to download datasets if it's the
first time you run our code. After that, your algorithm is running. Directory that stores datasets will be named as
``examples-da/unsupervised/data/<dataset name>``, and datasets for same settings(close set DA, partial DA, etc) are shared.

Step3: Analysis it
===================
``Dalib`` provides detailed intermediate results for you to monitor training process. You are recommended to watch these
results in a log file. You can find it in directory ``examples-da/unsupervised/logs/dann/Office31_A2W``.
In the ``txt`` file you can find results in following format::

    Epoch: [1][ 900/1000]	Time  0.60 ( 0.69)	Data  0.22 ( 0.31)	Loss   0.74 (  0.85)	Cls Acc 96.9 (95.1)	Domain Acc 64.1 (62.6)

Here your algorithm is running in epoch 2 (index starts from 0), has been trained for 900 iterations. The loss comes down to 0.74, on source domain
your classifier achieves 96.9% accuracy and your domain discriminator has an accuracy of 64.1%. Other algorithms may show you different statistics.

We provide you the most important statistics during training. But it's easy to print other information you care about through **modifying**
``dann.py``.

During training, we automatically save checkpoints for you. You can find ``latest`` and ``best`` model in directory ``examples-da/unsupervised/logs/dann/Office31_A2W/checkpoints``.
Resuming from checkpoint is also **supported**.
After training, you can test your algorithm's performance by passing in ``--phase test``, or you can visualize results by passing in ``--phase analysis``.
Corresponding commands are listed below:

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31
        -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --phase test

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31
        -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --phase analysis

Step4: Tune it
===================
We provide you hyper parameters settings corresponding to our reported results. Feel free to
change them by passing in different command-line arguments. Walking through ``dann.py`` will be helpful.
For example, in ``dann.py``, you can find following code:

.. code-block:: python

    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')

If you want to change dimension of your bottleneck layer to 512, you can simply pass in ``--bottleneck-dim 512``.

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31
        -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --bottleneck-dim 512

Also, we provide you the most important parameters to tune. Approaches such as adjusting network
structures will be covered in another section.
