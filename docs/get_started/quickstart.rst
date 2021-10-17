************
Quick Start
************

In this section, we will use `DANN <https://arxiv.org/abs/1505.07818>`_  as an example,
and show you how to reproduce the benchmark results.

Before going deeper, please **make sure** you have installed all the dependency.

Step1: Find it
===================
Our code for domain adaptation is in the directory ``examples/domain_adaptation``.

DANN is designed for close set domain adaptation tasks. You can find the training code in
``examples/domain_adaptation/image_classification``. This directory contains implementations for other algorithms such as ``CDAN``, ``MDD``.
For now, you only need to care about two files: ``dann.py`` and ``dann.sh``.

Step2: Run it
===================
``dann.py`` is an executable python file. And you can find all the necessary running scripts to
reproduce the benchmarks with specified hyper-parameters in ``dann.sh``.

For instance, running the following command will start training on ``Office-31`` dataset.

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W

Note that ``-s`` specifies the source domain, ``-t`` specifies the target domain,
and ``--log`` specifies where to store results.

After running the above command, it will download ``Office-31`` datasets from the Internet if it's the
first time you run the code. Directory that stores datasets will be named as
``examples/domain_adaptation/image_classification/data/<dataset name>``.

Step3: Analysis it
===================
If everything works fine, you will see results in following format::

    Epoch: [1][ 900/1000]	Time  0.60 ( 0.69)	Data  0.22 ( 0.31)	Loss   0.74 (  0.85)	Cls Acc 96.9 (95.1)	Domain Acc 64.1 (62.6)

``Trans-Learn`` provides detailed  statistics to monitor during training.

Here your algorithm is running in epoch 2 (index starts from 0), has been trained for 900 iterations.
The loss comes down to 0.74, on source domain your classifier achieves 96.9% accuracy
and your domain discriminator has an accuracy of 64.1%.
Different algorithms may show you different statistics.

You can also watch these results in a log file.
They are located in directory ``logs/dann/Office31_A2W/log.txt``.

During training, ``latest`` and ``best`` model checkpoints will be saved in directory ``logs/dann/Office31_A2W/checkpoints``.
Resuming from checkpoint is also **supported**.

After training, you can test your algorithm's performance by passing in ``--phase test``.

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --phase test

In next section, we will introduce how to visualize the results in details.