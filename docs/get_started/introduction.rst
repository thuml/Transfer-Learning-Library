************
Introduction
************

File Structure
===================

Currently, **Trans-Learn** provide algorithms for **domain adaptation (DA)** and **fine-tune**.
The training codes for domain adaptation and fine-tune is in directory ``examples-da`` and ``examples-ft`` respectively.

Since there are many settings or applications in domain adaptation, we divide them into different directories as follows.

===============================     ============================================
Directory                           Setting
examples-da/unsupervised            Unsupervised DA for Classification
examples-da/partial                 Partial DA for Classification
examples-da/open-set                Open Set DA for Classification
examples-da/multi-source            Multi-source DA for Classification
examples-da/regression              Unsupervised DA for Regression
examples-da/segmentation            Unsupervised DA for Segmentation
examples-da/keypoint_detection      Unsupervised DA for Keypoint Detection
===============================     ============================================

Besides training codes of different transfer learning algorithms, we also provide detailed API.

===============================     ==========================================================================
Directory                           Usage
common/vision                       Datasets, models, transforms frequently used in vision tasks.
common/utils                        Tools for training, analysis, or evaluation
common/modules                      Frequently used modules in transfer learning
dalib/adaptation                    Adaptation algorithms and losses for DA
dalib/translation                   Image translation algorithms for DA
dalib/modules                       Frequently used modules in DA
ftlib/finetune                      Finetune algorithms and losses
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





