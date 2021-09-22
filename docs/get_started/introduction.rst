************
Introduction
************

File Structure
===================

Currently, **Trans-Learn** provide algorithms for **domain adaptation (DA)** and **fine-tune**.
The training codes for domain adaptation and fine-tune is in directory ``examples``.

Since there are many settings or applications in transfer learning, we divide them into different directories as follows.

=============================================   ============================================
Directory                                       Setting
examples/domain_adaptation/unsupervised         Unsupervised DA for Classification
examples/domain_adaptation/partial              Partial DA for Classification
examples/domain_adaptation/openset              Open Set DA for Classification
examples/domain_adaptation/multi_source         Multi-source DA for Classification
examples/domain_adaptation/regression           Unsupervised DA for Regression
examples/domain_adaptation/segmentation         Unsupervised DA for Segmentation
examples/domain_adaptation/keypoint_detection   Unsupervised DA for Keypoint Detection
examples/finetune/classification                Finetune for Classification
=============================================   ============================================

Besides training codes of different transfer learning algorithms, we also provide detailed API.

===============================     ==========================================================================
Directory                           Usage
common/vision                       Datasets, models, transforms frequently used in vision tasks.
common/utils                        Tools for training, analysis, or evaluation
common/modules                      Frequently used modules in transfer learning
dalib/adaptation                    Adaptation algorithms and losses for DA
dalib/translation                   Image translation algorithms for DA
dalib/modules                       Frequently used modules in DA
talib/finetune                      Finetune algorithms and losses
===============================     ==========================================================================


How to Choose Algorithms?
======================================

Although there are many methods proposed for transfer learning each year, unfortunately, no method is universal.
Each method has its advantages and disadvantages and applicable scenarios.

Therefore, we provide benchmarks under different settings. They can serve as a reference when you choose an algorithm.

You are suggested to read related papers if you want to have a deeper understanding of an algorithm.

After completing the above things, trying is still the most important.
In addition to the final accuracy, you need to pay attention to the output of the program throughout the training process.
You can also use the tools in ``common.utils.analysis`` to visualize the results.





