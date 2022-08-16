# <img src="/logo.png" width=150/> Transfer Learning Library

- [Introduction](#introduction)
- [Updates](#updates)
- [Supported Methods](#supported-methods)
- [Installation](#installation)
- [Documentation](#documentation)
- [Contact](#contact)
- [Citation](#citation)

## Introduction
*TLlib* is an open-source and well-documented library for Transfer Learning. It is based on pure PyTorch with high performance and friendly API. Our code is pythonic, and the design is consistent with torchvision. You can easily develop new algorithms, or readily apply existing algorithms.

Our _API_ is divided by methods, which include: 
- domain alignment methods (tllib.aligment)
- domain translation methods (tllib.translation)
- self-training methods (tllib.self_training)
- regularization methods (tllib.regularization)
- data reweighting/resampling methods (tllib.reweight)
- model ranking/selection methods (tllib.ranking)
- normalization-based methods (tllib.normalization)

<img src="/Tllib.png">

We provide many example codes in the directory _examples_, which is divided by learning setups. Currently, the supported learning setups include:
- DA (domain adaptation)
- TA (task adaptation, also known as finetune)
- OOD (out-of-distribution generalization, also known as DG / domain generalization)
- SSL (semi-supervised learning)
- Model Selection 

Our supported tasks include: classification, regression, object detection, segmentation, keypoint detection, and so on.


## Updates 

### 2022.8
We release `v0.4` of *TLlib*. Previous versions of *TLlib* can be found [here](https://github.com/thuml/Transfer-Learning-Library/releases). In `v0.4`, we add implementations of 
the following methods:
- Domain Adaptation for Object Detection [[Code]](/examples/domain_adaptation/object_detection) [[API]](/tllib/alignment/d_adapt)
- Pre-trained Model Selection [[Code]](/examples/model_selection) [[API]](/tllib/ranking)
- Semi-supervised Learning for Classification [[Code]](/examples/semi_supervised_learning/image_classification/) [[API]](/tllib/self_training)

Besides, we maintain a collection of **_awesome papers in Transfer Learning_** in another repo [_A Roadmap for Transfer Learning_](https://github.com/thuml/A-Roadmap-for-Transfer-Learning).

### 2022.2
We adjusted our API following our survey [Transferablity in Deep Learning](https://arxiv.org/abs/2201.05867).

## Supported Methods
The currently supported algorithms include:

##### Domain Adaptation for Classification [[Code]](/examples/domain_adaptation/image_classification)
- **DANN** - Unsupervised Domain Adaptation by Backpropagation [[ICML 2015]](http://proceedings.mlr.press/v37/ganin15.pdf) [[Code]](/examples/domain_adaptation/image_classification/dann.py)
- **DAN** - Learning Transferable Features with Deep Adaptation Networks [[ICML 2015]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf) [[Code]](/examples/domain_adaptation/image_classification/dan.py)
- **JAN** - Deep Transfer Learning with Joint Adaptation Networks [[ICML 2017]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-adaptation-networks-icml17.pdf) [[Code]](/examples/domain_adaptation/image_classification/jan.py)
- **ADDA** - Adversarial Discriminative Domain Adaptation [[CVPR 2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf) [[Code]](/examples/domain_adaptation/image_classification/adda.py)
- **CDAN** - Conditional Adversarial Domain Adaptation [[NIPS 2018]](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) [[Code]](/examples/domain_adaptation/image_classification/cdan.py) 
- **MCD** - Maximum Classifier Discrepancy for Unsupervised Domain Adaptation [[CVPR 2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) [[Code]](/examples/domain_adaptation/image_classification/mcd.py)
- **MDD** - Bridging Theory and Algorithm for Domain Adaptation [[ICML 2019]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) [[Code]](/examples/domain_adaptation/image_classification/mdd.py) 
- **BSP** - Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation [[ICML 2019]](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) [[Code]](/examples/domain_adaptation/image_classification/bsp.py) 
- **MCC** - Minimum Class Confusion for Versatile Domain Adaptation [[ECCV 2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660460.pdf) [[Code]](/examples/domain_adaptation/image_classification/mcc.py)

##### Domain Adaptation for Object Detection [[Code]](/examples/domain_adaptation/object_detection)
- **CycleGAN** - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[ICCV 2017]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf) [[Code]](/examples/domain_adaptation/object_detection/cycle_gan.py)
- **D-adapt** - Decoupled Adaptation for Cross-Domain Object Detection [[ICLR 2022]](https://openreview.net/pdf?id=VNqaB1g9393) [[Code]](/examples/domain_adaptation/object_detection/d_adapt)

##### Domain Adaptation for Semantic Segmentation [[Code]](/examples/domain_adaptation/semantic_segmentation/)
- **CycleGAN** - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[ICCV 2017]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf) [[Code]](/examples/domain_adaptation/semantic_segmentation/cycle_gan.py)
- **CyCADA** - Cycle-Consistent Adversarial Domain Adaptation [[ICML 2018]](http://proceedings.mlr.press/v80/hoffman18a.html) [[Code]](/examples/domain_adaptation/semantic_segmentation/cycada.py)
- **ADVENT** - Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf) [[Code]](/examples/domain_adaptation/semantic_segmentation/advent.py)
- **FDA** - Fourier Domain Adaptation for Semantic Segmentation [[CVPR 2020]](https://arxiv.org/abs/2004.05498) [[Code]](/examples/domain_adaptation/semantic_segmentation/fda.py)

##### Domain Adaptation for Keypoint Detection [[Code]](/examples/domain_adaptation/keypoint_detection)
- **RegDA** - Regressive Domain Adaptation for Unsupervised Keypoint Detection [[CVPR 2021]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/regressive-domain-adaptation-cvpr21.pdf) [[Code]](/examples/domain_adaptation/keypoint_detection)

##### Domain Adaptation for Person Re-identification [[Code]](/examples/domain_adaptation/re_identification/)
- **IBN-Net** - Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net [[ECCV 2018]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf)
- **MMT** - Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification [[ICLR 2020]](https://arxiv.org/abs/2001.01526) [[Code]](/examples/domain_adaptation/re_identification/mmt.py)
- **SPGAN** - Similarity Preserving Generative Adversarial Network [[CVPR 2018]](https://arxiv.org/pdf/1811.10551.pdf) [[Code]](/examples/domain_adaptation/re_identification/spgan.py)

##### Partial Domain Adaptation [[Code]](/examples/domain_adaptation/partial_domain_adaptation)
- **IWAN** - Importance Weighted Adversarial Nets for Partial Domain Adaptation[[CVPR 2018]](https://arxiv.org/abs/1803.09210) [[Code]](/examples/domain_adaptation/partial_domain_adaptation/iwan.py)
- **AFN** - Larger Norm More Transferable: An Adaptive Feature Norm Approach for
Unsupervised Domain Adaptation [[ICCV 2019]](https://arxiv.org/pdf/1811.07456v2.pdf) [[Code]](/examples/domain_adaptation/partial_domain_adaptation/afn.py)

##### Open-set Domain Adaptation [[Code]](/examples/domain_adaptation/openset_domain_adaptation)
- **OSBP** - Open Set Domain Adaptation by Backpropagation [[ECCV 2018]](https://arxiv.org/abs/1804.10427) [[Code]](/examples/domain_adaptation/openset_domain_adaptation/osbp.py)

##### Domain Generalization for Classification [[Code]](/examples/domain_generalization/image_classification/)
- **IBN-Net** - Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net [[ECCV 2018]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf)
- **MixStyle** - Domain Generalization with MixStyle [[ICLR 2021]](https://arxiv.org/abs/2104.02008) [[Code]](/examples/domain_generalization/image_classification/mixstyle.py)
- **MLDG** - Learning to Generalize: Meta-Learning for Domain Generalization [[AAAI 2018]](https://arxiv.org/pdf/1710.03463.pdf) [[Code]](/examples/domain_generalization/image_classification/mldg.py)
- **IRM** - Invariant Risk Minimization [[ArXiv]](https://arxiv.org/abs/1907.02893) [[Code]](/examples/domain_generalization/image_classification/irm.py)
- **VREx** - Out-of-Distribution Generalization via Risk Extrapolation [[ICML 2021]](https://arxiv.org/abs/2003.00688) [[Code]](/examples/domain_generalization/image_classification/vrex.py)
- **GroupDRO** - Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization [[ArXiv]](https://arxiv.org/abs/1911.08731) [[Code]](/examples/domain_generalization/image_classification/groupdro.py)
- **Deep CORAL** - Correlation Alignment for Deep Domain Adaptation [[ECCV 2016]](https://arxiv.org/abs/1607.01719) [[Code]](/examples/domain_generalization/image_classification/coral.py)

##### Domain Generalization for Person Re-identification [[Code]](/examples/domain_generalization/re_identification/)
- **IBN-Net** - Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net [[ECCV 2018]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf)
- **MixStyle** - Domain Generalization with MixStyle [[ICLR 2021]](https://arxiv.org/abs/2104.02008) [[Code]](/examples/domain_generalization/re_identification/mixstyle.py)

##### Task Adaptation (Fine-Tuning) for Image Classification [[Code]](/examples/task_adaptation/image_classification/)
- **L2-SP** - Explicit inductive bias for transfer learning with convolutional networks [[ICML 2018]]((https://arxiv.org/abs/1802.01483)) [[Code]](/examples/task_adaptation/image_classification/delta.py)
- **BSS** - Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning [[NIPS 2019]](https://proceedings.neurips.cc/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf) [[Code]](/examples/task_adaptation/image_classification/bss.py)
- **DELTA** - DEep Learning Transfer using Fea- ture Map with Attention for convolutional networks [[ICLR 2019]](https://openreview.net/pdf?id=rkgbwsAcYm) [[Code]](/examples/task_adaptation/image_classification/delta.py)
- **Co-Tuning** - Co-Tuning for Transfer Learning [[NIPS 2020]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/co-tuning-for-transfer-learning-nips20.pdf) [[Code]](/examples/task_adaptation/image_classification/co_tuning.py)
- **StochNorm** - Stochastic Normalization [[NIPS 2020]](https://papers.nips.cc/paper/2020/file/bc573864331a9e42e4511de6f678aa83-Paper.pdf) [[Code]](/examples/task_adaptation/image_classification/stochnorm.py)
- **LWF** - Learning Without Forgetting [[ECCV 2016]](https://arxiv.org/abs/1606.09282) [[Code]](/examples/task_adaptation/image_classification/lwf.py)
- **Bi-Tuning** - Bi-tuning of Pre-trained Representations [[ArXiv]](https://arxiv.org/abs/2011.06182?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29) [[Code]](/examples/task_adaptation/image_classification/bi_tuning.py)

##### Pre-trained Model Selection [[Code]](/examples/model_selection)

- **H-Score** - An Information-theoretic Approach to Transferability in Task Transfer Learning [[ICIP 2019]](http://yangli-feasibility.com/home/media/icip-19.pdf) [[Code]](/examples/model_selection/hscore.py)
- **NCE** - Negative Conditional Entropy in `Transferability and Hardness of Supervised Classification Tasks [[ICCV 2019]](https://arxiv.org/pdf/1908.08142v1.pdf) [[Code]](/examples/model_selection/nce.py)
- **LEEP** - LEEP: A New Measure to Evaluate Transferability of Learned Representations [[ICML 2020]](http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf) [[Code]](/examples/model_selection/leep.py)
- **LogME** - Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models for Transfer Learning [[ICML 2021]](https://arxiv.org/pdf/2102.11005.pdf) [[Code]](/examples/model_selection/logme.py)

##### Semi-Supervised Learning for Classification [[Code]](/examples/semi_supervised_learning/image_classification/)
- **Pseudo Label** - Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks [[ICML 2013]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf) [[Code]](/examples/semi_supervised_learning/image_classification/pseudo_label.py)
- **Pi Model** - Temporal Ensembling for Semi-Supervised Learning [[ICLR 2017]](https://arxiv.org/abs/1610.02242) [[Code]](/examples/semi_supervised_learning/image_classification/pi_model.py)
- **Mean Teacher** - Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results [[NIPS 2017]](https://arxiv.org/abs/1703.01780) [[Code]](/examples/semi_supervised_learning/image_classification/mean_teacher.py)
- **Noisy Student** - Self-Training With Noisy Student Improves ImageNet Classification [[CVPR 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf) [[Code]](/examples/semi_supervised_learning/image_classification/noisy_student.py)
- **UDA** - Unsupervised Data Augmentation for Consistency Training [[NIPS 2020]](https://arxiv.org/pdf/1904.12848v4.pdf) [[Code]](/examples/semi_supervised_learning/image_classification/uda.py)
- **FixMatch** - Simplifying Semi-Supervised Learning with Consistency and Confidence [[NIPS 2020]](https://arxiv.org/abs/2001.07685) [[Code]](/examples/semi_supervised_learning/image_classification/fixmatch.py)
- **Self-Tuning** - Self-Tuning for Data-Efficient Deep Learning [[ICML 2021]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/Self-Tuning-for-Data-Efficient-Deep-Learning-icml21.pdf) [[Code]](/examples/semi_supervised_learning/image_classification/self_tuning.py)
- **FlexMatch** - FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling [[NIPS 2021]](https://arxiv.org/abs/2110.08263) [[Code]](/examples/semi_supervised_learning/image_classification/flexmatch.py)
- **DebiasMatch** - Debiased Learning From Naturally Imbalanced Pseudo-Labels [[CVPR 2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Debiased_Learning_From_Naturally_Imbalanced_Pseudo-Labels_CVPR_2022_paper.pdf) [[Code]](/examples/semi_supervised_learning/image_classification/debiasmatch.py)
- **DST** - Debiased Self-Training for Semi-Supervised Learning [[ArXiv]](https://arxiv.org/abs/2202.07136) [[Code]](/examples/semi_supervised_learning/image_classification/dst.py)

## Installation

- To use ``tllib`` in other places, you need to install TLlib,
```shell
python setup.py install
```
Note that we do not support *pip install* currently.

- For flexible use and modification of TLlib, please git clone the library and check that you have install all the dependency.

```
    pip install -r requirements.txt
```

It's recommended to use pytorch==1.7.1 and torchvision==0.8.2 in order to better reproduce the benchmark results.


## Documentation
You can find the API documentation on the website: [Documentation](http://tl.thuml.ai/).

## Usage
You can find examples in the directory `examples`. A typical usage is 
```shell script
# Train a DANN on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
python dann.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 20
```

## Contributing
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. 

## Disclaimer on Datasets

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have licenses to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!


## Contact
If you have any problem with our code or have some suggestions, including the future feature, feel free to contact 
- Junguang Jiang (JiangJunguang1123@outlook.com)
- Baixu Chen (cbx_99_hasta@outlook.com)
- Mingsheng Long (longmingsheng@gmail.com)

or describe it in Issues.

For Q&A in Chinese, you can choose to ask questions here before sending an email. [迁移学习算法库答疑专区](https://zhuanlan.zhihu.com/p/248104070)

## Citation

If you use this toolbox or benchmark in your research, please cite this project. 

```latex
@misc{jiang2022transferability,
      title={Transferability in Deep Learning: A Survey}, 
      author={Junguang Jiang and Yang Shu and Jianmin Wang and Mingsheng Long},
      year={2022},
      eprint={2201.05867},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{tllib,
    author = {Junguang Jiang, Baixu Chen, Bo Fu, Mingsheng Long},
    title = {Transfer-Learning-library},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/thuml/Transfer-Learning-Library}},
}
```

## Acknowledgment

We would like to thank School of Software, Tsinghua University and The National Engineering Laboratory for Big Data Software for providing such an excellent ML research platform.

