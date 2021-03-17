<img src="https://github.com/thuml/Transfer-Learning-Library/blob/dev/TransLearn.png"/>

## Introduction

*Trans-Learn* is a Transfer Learning library based on pure PyTorch with high performance and friendly API. 
Our code is pythonic, and the design is consistent with torchvision. You can easily develop new algorithms, or easily apply existing algorithms..

This is the development branch for *Trans-Learn*. 
Compared with the master version, we have added

- Regression DA （including Source Only, DD)
- Unsupervised DA (including MCC, AFN)
- Partial DA (DANN, PADA, IWAN)
- Open Set DA (DANN, OSBP)
- Segmentation DA (ADVENT, FDA, CycleGAN, Cycada)
- Keypoint Detection DA (RegDA)

We are planning to add
- Segmentation DA (Self-training methods)
- Finetune Library (ftlib)
- Object Detection DA

The performance of these algorithms were fairly evaluated in this [benchmark](http://microhhh.com/dalib/index.html).

There might be many errors and changes in this branch. Please refer [master](https://github.com/thuml/Transfer-Learning-Library) for stable version. Also, any suggestions are welcome!

## Installation

For flexible use and modification, please git clone the library.

## Documentation
You can find the tutorial and API documentation on the website: [Documentation (please open in Firefox or Safari)](http://microhhh.com/). Note that this link is only for temporary use. You can also build the doc by yourself following the instructions in http://microhhh.com/get_started/faq.html.

In the directory `examples-da` and `examples-ft`, you can find all the necessary running scripts to reproduce the benchmarks with specified hyper-parameters.

## Contributing
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. 


## Disclaimer on Datasets

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have licenses to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!


## Contact
If you have any problem with our code or have some suggestions, including the future feature, feel free to contact 
- Junguang Jiang (JiangJunguang1123@outlook.com)
- Bo Fu (fb1121@vip.qq.com)
- Mingsheng Long (longmingsheng@gmail.com)

or describe it in Issues.

For Q&A in Chinese, you can choose to ask questions here before sending an email. [迁移学习算法库答疑专区](https://zhuanlan.zhihu.com/p/248104070)

## Citation

If you use this toolbox or benchmark in your research, please cite this project. 

```latex
@misc{dalib,
  author = {Junguang Jiang, Bo Fu, Mingsheng Long},
  title = {Transfer-Learning-library},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thuml/Transfer-Learning-Library}},
}
```

## Acknowledgment

We would like to thank School of Software, Tsinghua University and The National Engineering Laboratory for Big Data Software for providing such an excellent ML research platform.

