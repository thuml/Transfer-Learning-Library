************
Introduction
************

Currently, **Trans-Learn** provide algorithms for **domain adaptation (DA)**, **task adaptation (TA)**, and **domain generalization (DG)**.
We implement training codes (in directory ``examples``) in a unified way, which allows quantitative, fair, and reproducible comparisons between different algorithms.


Comparison with Existing Library
===================================

There exist several open-source transfer learning libraries,
but most of them only focus on domain adaptation, especially statistics matching and adversarial adaptation, while ignoring other adaptation methods.

In contrast, Transfer-Learn covers more types of adaptation algorithms.

- For more convenient  algorithm selection, we provide the most comprehensive benchmark among all those libraries.
- For faster algorithm reproduction, we provide training scripts for all the results in the benchmark.
- For better application of the library, we provide detailed documentation and examples in many downstream tasks.

As far as we know, Transfer-Learn is currently the only one open source transfer learning library that can meet all the above requirements.

=========================== =================   ====================    =============== ==============  ======================
Features                    Transfer-Learn      transferlearning.xyz    salad           Dassl           deep transfer learning
Task Adaptation             :math:`\surd`       :math:`\times`          :math:`\times`  :math:`\times`  :math:`\times`
Statistics Matching         :math:`\surd`       :math:`\surd`           :math:`\surd`   :math:`\surd`   :math:`\surd`
Adversarial Adaptation      :math:`\surd`       :math:`\surd`           :math:`\surd`   :math:`\surd`   :math:`\surd`
Disparity Discrepancy       :math:`\surd`       :math:`\times`          :math:`\times`  :math:`\times`  :math:`\times`
Translation                 :math:`\surd`       :math:`\times`          :math:`\times`  :math:`\times`  :math:`\times`
Benchmark                   :math:`\surd`       :math:`\surd`           :math:`\surd`   :math:`\times`  :math:`\surd`
Reproducible Scripts        :math:`\surd`       :math:`\surd`           :math:`\surd`   :math:`\times`  :math:`\times`
Detailed Documentation      :math:`\surd`       :math:`\times`          :math:`\surd`   :math:`\times`  :math:`\times`
Application                 :math:`\surd`       :math:`\times`          :math:`\times`  :math:`\times`  :math:`\surd`
=========================== =================   ====================    =============== ==============  ======================

Library Usage
=============
Here we give a short description on how to use Transfer-Learn using `DANN <https://arxiv.org/abs/1505.07818>`_ as an instance.

In the original implementation of DANN, the domain adversarial loss, domain discriminator, feature extractor, and classifier are tightly coupled together in one *nn.Module*, which cause the difficulty of reuse, e.g., the entire algorithm needs re-implementation when the input data is changed from image to text. Yet in this case, the domain adversarial loss and the  domain discriminator remains unchanged and can be reused.

Thus, in Transfer-Learn, models and loss functions are decoupled.
When using DANN, users need to initialize a domain discriminator and pass it to the domain adversarial loss module, and then use this module the same way as  the cross-entropy loss module defined in the Pytorch,


.. code:: python

    >>> # define the domain discriminator
    >>> from dalib.modules.domain_discriminator import DomainDiscriminator
    >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
    >>> # define the domain adversarial loss module
    >>> from dalib.adptation.dann import DomainAdversarialLoss
    >>> dann = DomainAdversarialLoss(discriminator, reduction='mean')
    >>> # features from the source and target domain
    >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
    >>> # calculate the final loss
    >>> loss = dann(f_s, f_t)


Transfer-Learn provide similar API for other transfer learning algorithms. Detailed usages of these algorithms can be found at the documentation.

Library philosophy
=====================

Transfer-Learn is designed to be *extensible* by researchers and *simple* for practitioners.
Currently, there are mainly two types of algorithm implementations.

- One is to encapsulate each algorithm into a Trainer, whose typical representative is *Pytorch-lighting*. The user only needs to feed the training data to it, and does not need to care about the specific training process.
- Another strategy is to encapsulate the core loss function in each algorithm, and the user needs to implement the complete training process themselves. A typical representative is *Pytorch*.

Although the former method is easier to use, it is less extensible. Since it's often necessary to adjust training process in different transfer learning scenarios, Transfer-Learn adopts the latter method for better extensibility.

Following Pytorch, we provide training examples of different transfer algorithms in different scenarios, which allows users to quickly adapt to Transfer-Learn as long as they have learned Pytorch before.

We try our best to make Transfer-Learn easy to start with, e.g., we support the automatic download of most common transfer learning datasets so that users do not need to spend time on data preprocessing.


How to Choose Algorithms?
======================================

Although there are many methods proposed for transfer learning each year, unfortunately, no method is universal.
Each method has its advantages and disadvantages and applicable scenarios.

Therefore, we provide benchmarks under different settings. They can serve as a reference when you choose an algorithm.

You are suggested to read related papers if you want to have a deeper understanding of an algorithm.

After completing the above things, trying is still the most important.
In addition to the final accuracy, you need to pay attention to the output of the program throughout the training process.
You can also use the tools in ``common.utils.analysis`` to visualize the results.





