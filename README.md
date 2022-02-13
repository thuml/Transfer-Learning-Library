<img src="https://github.com/thuml/Transfer-Learning-Library/blob/master/TransLearn.png"/>

## Introduction
*TLlib* is an open-source and well-documented library for Transfer Learning. It is based on pure PyTorch with high performance and friendly API. Our code is pythonic, and the design is consistent with torchvision. You can easily develop new algorithms, or readily apply existing algorithms.

The currently supported algorithms include:

##### [Domain Adaptation for Classification](/examples/domain_adaptation/image_classification)
- **DANN** - Unsupervised Domain Adaptation by Backpropagation [[ICML 2015]](http://proceedings.mlr.press/v37/ganin15.pdf) [[Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/dann.py)
- **DAN** - Learning Transferable Features with Deep Adaptation Networks [[ICML 2015]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf) [[Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/dan.py)
- **JAN** - Deep Transfer Learning with Joint Adaptation Networks [[ICML 2017]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-adaptation-networks-icml17.pdf) [[Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/jan.py)
- **ADDA** - Adversarial Discriminative Domain Adaptation [[CVPR2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf) [[Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/adda.py)
- **CDAN** - Conditional Adversarial Domain Adaptation [[NIPS 2018]](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) [[Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/cdan.py) 
- **MCD** - Maximum Classifier Discrepancy for Unsupervised Domain Adaptation [[CVPR 2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) [[Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/mcd.py)
- **MDD** - Bridging Theory and Algorithm for Domain Adaptation [[ICML 2019]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) [[Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/mdd.py) 
- **BSP** - Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation [[ICML 2019]](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) [[Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/bsp.py) 
- **MCC** - Minimum Class Confusion for Versatile Domain Adaptation [[ECCV 2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660460.pdf) [[Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/mcc.py)


##### [Partial Domain Adaptation](/examples/domain_adaptation/partial_domain_adaptation/)
- [Domain Adversarial Neural Network (DANN)](https://arxiv.org/abs/1505.07818)
- [Partial Adversarial Domain Adaptation (PADA)](https://arxiv.org/abs/1808.04205)
- [Importance Weighted Adversarial Nets (IWAN)](https://arxiv.org/abs/1803.09210)
- [Adaptive Feature Norm (AFN)](https://arxiv.org/pdf/1811.07456v2.pdf)

##### [Open-set Domain Adaptation](/examples/domain_adaptation/openset_domain_adaptation/)
- [Open Set Domain Adaptation (OSBP)](https://arxiv.org/abs/1804.10427)

##### [Domain Adaptation for Semantic Segmentation](/examples/domain_adaptation/semantic_segmentation/)
- [Cycle-Consistent Adversarial Networks (CycleGAN)](https://arxiv.org/pdf/1703.10593.pdf)
- [CyCADA: Cycle-Consistent Adversarial Domain Adaptation](https://arxiv.org/abs/1711.03213)
- [Adversarial Entropy Minimization (ADVENT)](https://arxiv.org/abs/1811.12833)
- [Fourier Domain Adaptation (FDA)](https://arxiv.org/abs/2004.05498)

##### [Domain Adaptation for Keypoint Detection](/examples/domain_adaptation/keypoint_detection)
- [Regressive Domain Adaptation for Unsupervised Keypoint Detection (RegDA, CVPR 2021)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/regressive-domain-adaptation-cvpr21.pdf)

##### [Domain Adaptation for Person Re-identification](/examples/domain_adaptation/re_identification/)
- [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net (IBN-Net, ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf)
- [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification (MMT, ICLR 2020)](https://arxiv.org/abs/2001.01526)
- [Similarity Preserving Generative Adversarial Network (SPGAN, CVPR 2018)](https://arxiv.org/pdf/1811.10551.pdf)

##### [Task Adaptation for Image Classification](/examples/task_adaptation/image_classification/)
- [Explicit inductive bias for transfer learning with convolutional networks
    (L2-SP, ICML 2018)](https://arxiv.org/abs/1802.01483)
- [Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning (BSS, NIPS 2019)](https://proceedings.neurips.cc/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf)
- [DEep Learning Transfer using Fea- ture Map with Attention for convolutional networks (DELTA, ICLR 2019)](https://openreview.net/pdf?id=rkgbwsAcYm)
- [Co-Tuning for Transfer Learning (Co-Tuning, NIPS 2020)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/co-tuning-for-transfer-learning-nips20.pdf)
- [Stochastic Normalization (StochNorm, NIPS 2020)](https://papers.nips.cc/paper/2020/file/bc573864331a9e42e4511de6f678aa83-Paper.pdf)
- [Learning Without Forgetting (LWF, ECCV 2016)](https://arxiv.org/abs/1606.09282)
- [Bi-tuning of Pre-trained Representations (Bi-Tuning)](https://arxiv.org/abs/2011.06182?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)

##### [Domain Generalization for Classification](/examples/domain_generalization/image_classification/)
- [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net (IBN-Net, ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf)
- [Domain Generalization with MixStyle (MixStyle, ICLR 2021)](https://arxiv.org/abs/2104.02008)
- [Learning to Generalize: Meta-Learning for Domain Generalization (MLDG, AAAI 2018)](https://arxiv.org/pdf/1710.03463.pdf)
- [Invariant Risk Minimization (IRM)](https://arxiv.org/abs/1907.02893)
- [Out-of-Distribution Generalization via Risk Extrapolation (VREx, ICML 2021)](https://arxiv.org/abs/2003.00688)
- [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization(GroupDRO)](https://arxiv.org/abs/1911.08731)
- [Deep CORAL: Correlation Alignment for Deep Domain Adaptation (Deep Coral, ECCV 2016)](https://arxiv.org/abs/1607.01719)

##### [Domain Generalization for Person Re-identification](/examples/domain_generalization/re_identification/)
- [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net (IBN-Net, ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf)
- [Domain Generalization with MixStyle (MixStyle, ICLR 2021)](https://arxiv.org/abs/2104.02008)


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

