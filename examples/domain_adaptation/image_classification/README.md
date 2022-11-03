# Unsupervised Domain Adaptation for Image Classification

## Installation

It’s suggested to use **pytorch==1.7.1** and torchvision==0.8.2 in order to reproduce the benchmark results.

Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models). You
also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [MNIST](http://yann.lecun.com/exdb/mnist/), [SVHN](http://ufldl.stanford.edu/housenumbers/)
  , [USPS](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps)
- [Office31](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeCaltech](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [VisDA2017](http://ai.bu.edu/visda-2017/)
- [DomainNet](http://ai.bu.edu/M3SDA/)

You need to prepare following datasets manually if you want to use them:

- [ImageNet](https://www.image-net.org/)
- [ImageNetR](https://github.com/hendrycks/imagenet-r)
- [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

and prepare them following [Documentation for ImageNetR](/common/vision/datasets/imagenet_r.py)
and [ImageNet-Sketch](/common/vision/datasets/imagenet_sketch.py).

## Supported Methods

Supported methods include:

- [Domain Adversarial Neural Network (DANN)](https://arxiv.org/abs/1505.07818)
- [Deep Adaptation Network (DAN)](https://arxiv.org/pdf/1502.02791)
- [Joint Adaptation Network (JAN)](https://arxiv.org/abs/1605.06636)
- [Adversarial Discriminative Domain Adaptation (ADDA)](https://arxiv.org/pdf/1702.05464.pdf)
- [Conditional Domain Adversarial Network (CDAN)](https://arxiv.org/abs/1705.10667)
- [Maximum Classifier Discrepancy (MCD)](https://arxiv.org/abs/1712.02560)
- [Adaptive Feature Norm (AFN)](https://arxiv.org/pdf/1811.07456v2.pdf)
- [Batch Spectral Penalization (BSP)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/batch-spectral-penalization-icml19.pdf)
- [Margin Disparity Discrepancy (MDD)](https://arxiv.org/abs/1904.05801)
- [Minimum Class Confusion (MCC)](https://arxiv.org/abs/1912.03699)
- [FixMatch](https://arxiv.org/abs/2001.07685)

## Usage

The shell files give the script to reproduce the benchmark with specified hyper-parameters. For example, if you want to
train DANN on Office31, use the following script

```shell script
# Train a DANN on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
```

Note that ``-s`` specifies the source domain, ``-t`` specifies the target domain, and ``--log`` specifies where to store
results.

After running the above command, it will download ``Office-31`` datasets from the Internet if it's the first time you
run the code. Directory that stores datasets will be named as
``examples/domain_adaptation/image_classification/data/<dataset name>``.

If everything works fine, you will see results in following format::

    Epoch: [1][ 900/1000]	Time  0.60 ( 0.69)	Data  0.22 ( 0.31)	Loss   0.74 (  0.85)	Cls Acc 96.9 (95.1)	Domain Acc 64.1 (62.6)

You can also watch these results in the log file ``logs/dann/Office31_A2W/log.txt``.

After training, you can test your algorithm's performance by passing in ``--phase test``.

```
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --phase test
```

## Experiment and Results

**Notations**

- ``Origin`` means the accuracy reported by the original paper.
- ``Avg`` is the accuracy reported by `TLlib`.
- ``ERM`` refers to the model trained with data from the source domain.
- ``Oracle`` refers to the model trained with data from the target domain.

We found that the accuracies of adversarial methods (including DANN, ADDA, CDAN, MCD, BSP and MDD) are not stable even
after the random seed is fixed, thus we repeat running adversarial methods on *Office-31* and *VisDA-2017*
for three times and report their average accuracy.

### Office-31 accuracy on ResNet-50

| Methods | Origin | Avg  | A → W | D → W | W → D | A → D | D → A | W → A |
|---------|--------|------|-------|-------|-------|-------|-------|-------|
| ERM     | 76.1   | 79.5 | 75.8  | 95.5  | 99.0  | 79.3  | 63.6  | 63.8  |
| DANN    | 82.2   | 86.1 | 91.4  | 97.9  | 100.0 | 83.6  | 73.3  | 70.4  |
| ADDA    | /      | 87.3 | 94.6  | 97.5  | 99.7  | 90.0  | 69.6  | 72.5  |
| BSP     | 87.7   | 87.8 | 92.7  | 97.9  | 100.0 | 88.2  | 74.1  | 73.8  |
| DAN     | 80.4   | 83.7 | 84.2  | 98.4  | 100.0 | 87.3  | 66.9  | 65.2  |
| JAN     | 84.3   | 87.0 | 93.7  | 98.4  | 100.0 | 89.4  | 69.2  | 71.0  |
| CDAN    | 87.7   | 87.7 | 93.8  | 98.5  | 100.0 | 89.9  | 73.4  | 70.4  |
| MCD     | /      | 85.4 | 90.4  | 98.5  | 100.0 | 87.3  | 68.3  | 67.6  |
| AFN     | 85.7   | 88.6 | 94.0  | 98.9  | 100.0 | 94.4  | 72.9  | 71.1  |
| MDD     | 88.9   | 89.6 | 95.6  | 98.6  | 100.0 | 94.4  | 76.6  | 72.2  |
| MCC     | 89.4   | 89.6 | 94.1  | 98.4  | 99.8  | 95.6  | 75.5  | 74.2  |
| FixMatch| /      | 86.4 | 86.4  | 98.2  | 100.0 | 95.4  | 70.0  | 68.1  |

### Office-Home accuracy on ResNet-50

| Methods     | Origin | Avg  | Ar → Cl | Ar → Pr | Ar → Rw | Cl → Ar | Cl → Pr | Cl → Rw | Pr → Ar | Pr → Cl | Pr → Rw | Rw → Ar | Rw → Cl | Rw → Pr |
|-------------|--------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| ERM         | 46.1   | 58.4 | 41.1    | 65.9    | 73.7    | 53.1    | 60.1    | 63.3    | 52.2    | 36.7    | 71.8    | 64.8    | 42.6    | 75.2    |
| DAN         | 56.3   | 61.4 | 45.6    | 67.7    | 73.9    | 57.7    | 63.8    | 66.0    | 54.9    | 40.0    | 74.5    | 66.2    | 49.1    | 77.9    |
| DANN        | 57.6   | 65.2 | 53.8    | 62.6    | 74.0    | 55.8    | 67.3    | 67.3    | 55.8    | 55.1    | 77.9    | 71.1    | 60.7    | 81.1    |
| ADDA        | /      | 65.6 | 52.6    | 62.9    | 74.0    | 59.7    | 68.0    | 68.8    | 61.4    | 52.5    | 77.6    | 71.1    | 58.6    | 80.2    |
| JAN         | 58.3   | 65.9 | 50.8    | 71.9    | 76.5    | 60.6    | 68.3    | 68.7    | 60.5    | 49.6    | 76.9    | 71.0    | 55.9    | 80.5    |
| CDAN        | 65.8   | 68.8 | 55.2    | 72.4    | 77.6    | 62.0    | 69.7    | 70.9    | 62.4    | 54.3    | 80.5    | 75.5    | 61.0    | 83.8    |
| MCD         | /      | 67.8 | 51.7    | 72.2    | 78.2    | 63.7    | 69.5    | 70.8    | 61.5    | 52.8    | 78.0    | 74.5    | 58.4    | 81.8    |
| BSP         | 64.9   | 67.6 | 54.7    | 67.7    | 76.2    | 61.0    | 69.4    | 70.9    | 60.9    | 55.2    | 80.2    | 73.4    | 60.3    | 81.2    |
| AFN         | 67.3   | 68.2 | 53.2    | 72.7    | 76.8    | 65.0    | 71.3    | 72.3    | 65.0    | 51.4    | 77.9    | 72.3    | 57.8    | 82.4    |
| MDD         | 68.1   | 69.7 | 56.2    | 75.4    | 79.6    | 63.5    | 72.1    | 73.8    | 62.5    | 54.8    | 79.9    | 73.5    | 60.9    | 84.5    |
| MCC         | /      | 72.4 | 58.4    | 79.6    | 83.0    | 67.5    | 77.0    | 78.5    | 66.6    | 54.8    | 81.8    | 74.4    | 61.4    | 85.6    |
| FixMatch    | /      | 70.8 | 56.4    | 76.4    | 79.9    | 65.3    | 73.8    | 71.2    | 67.2    | 56.4    | 80.6    | 74.9    | 63.5    | 84.3    |

### Office-Home accuracy on vit_base_patch16_224 (batch size 24)

| Methods     | Ar → Cl | Ar → Pr | Ar → Rw | Cl → Ar | Cl → Pr | Cl → Rw | Pr → Ar | Pr → Cl | Pr → Rw | Rw → Ar | Rw → Cl | Rw → Pr | Avg  |
|-------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|------|
| Source Only | 52.4    | 82.1    | 86.9    | 76.8    | 84.1    | 86      | 75.1    | 51.2    | 88.1    | 78.3    | 51.5    | 87.8    | 75.0 |
| DANN        | 60.1    | 80.8    | 87.9    | 78.1    | 82.6    | 85.9    | 78.8    | 63.2    | 90.2    | 82.3    | 64      | 89.3    | 78.6 |
| DAN         | 56.3    | 83.6    | 87.5    | 77.7    | 84.7    | 86.7    | 75.9    | 54.5    | 88.5    | 80.2    | 56.2    | 88.2    | 76.7 |
| JAN         | 60.1    | 86.9    | 88.6    | 79.2    | 85.4    | 86.7    | 80.4    | 59.4    | 89.6    | 82      | 60.7    | 89.9    | 79.1 |
| CDAN        | 61.6    | 87.8    | 89.6    | 81.4    | 88.1    | 88.5    | 82.4    | 62.5    | 90.8    | 84.2    | 63.5    | 90.8    | 80.9 |
| MCD         | 52.3    | 75.3    | 85.3    | 75.4    | 75.4    | 78.3    | 68.8    | 49.7    | 86      | 80.6    | 60      | 89      | 73.0 |
| AFN         | 58.3    | 87.2    | 88.2    | 81.7    | 87      | 88.2    | 81      | 58.4    | 89.2    | 81.5    | 59.2    | 89.2    | 79.1 |
| MDD         | 64      | 89.3    | 90.4    | 82.2    | 87.7    | 89.2    | 82.8    | 64.9    | 91.7    | 83.7    | 65.4    | 92      | 81.9 |

### VisDA-2017 accuracy ResNet-101

| Methods     | Origin | Mean | plane | bcycl | bus  | car  | horse | knife | mcycl | person | plant | sktbrd | train | truck | Avg  |
|-------------|--------|------|-------|-------|------|------|-------|-------|-------|--------|-------|--------|-------|-------|------|
| ERM         | 52.4   | 51.7 | 63.6  | 35.3  | 50.6 | 78.2 | 74.6  | 18.7  | 82.1  | 16.0   | 84.2  | 35.5   | 77.4  | 4.7   | 56.9 |
| DANN        | 57.4   | 79.5 | 93.5  | 74.3  | 83.4 | 50.7 | 87.2  | 90.2  | 89.9  | 76.1   | 88.1  | 91.4   | 89.7  | 39.8  | 74.9 |
| ADDA        | /      | 77.5 | 95.6  | 70.8  | 84.4 | 54.0 | 87.8  | 75.8  | 88.4  | 69.3   | 84.1  | 86.2   | 85.0  | 48.0  | 74.3 |
| BSP         | 75.9   | 80.5 | 95.7  | 75.6  | 82.8 | 54.5 | 89.2  | 96.5  | 91.3  | 72.2   | 88.9  | 88.7   | 88.0  | 43.4  | 76.2 |
| DAN         | 61.1   | 66.4 | 89.2  | 37.2  | 77.7 | 61.8 | 81.7  | 64.3  | 90.6  | 61.4   | 79.9  | 37.7   | 88.1  | 27.4  | 67.2 |
| JAN         | /      | 73.4 | 96.3  | 66.0  | 82.0 | 44.1 | 86.4  | 70.3  | 87.9  | 74.6   | 83.0  | 64.6   | 84.5  | 41.3  | 70.3 |
| CDAN        | /      | 80.1 | 94.0  | 69.2  | 78.9 | 57.0 | 89.8  | 94.9  | 91.9  | 80.3   | 86.8  | 84.9   | 85.0  | 48.5  | 76.5 |
| MCD         | 71.9   | 77.7 | 87.8  | 75.7  | 84.2 | 78.1 | 91.6  | 95.3  | 88.1  | 78.3   | 83.4  | 64.5   | 84.8  | 20.9  | 76.7 |
| AFN         | 76.1   | 75.0 | 95.6  | 56.2  | 81.3 | 69.8 | 93.0  | 81.0  | 93.4  | 74.1   | 91.7  | 55.0   | 90.6  | 18.1  | 74.4 |
| MDD         | /      | 82.0 | 88.3  | 62.8  | 85.2 | 69.9 | 91.9  | 95.1  | 94.4  | 81.2   | 93.8  | 89.8   | 84.1  | 47.9  | 79.8 |
| MCC         | 78.8   | 83.6 | 95.3  | 85.8  | 77.1 | 68.0 | 93.9  | 92.9  | 84.5  | 79.5   | 93.6  | 93.7   | 85.3  | 53.8  | 80.4 |
| FixMatch    | /      | 79.5 | 96.5  | 76.6  | 72.6 | 84.6 | 96.3  | 92.6  | 90.5  | 81.8   | 91.9  | 74.6   | 87.3  | 8.6   | 78.4 |

### DomainNet accuracy on ResNet-101

| Methods   | c->p | c->r | c->s | p->c | p->r | p->s | r->c | r->p | r->s | s->c | s->p | s->r | Avg  |
|-------------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| ERM         | 32.7 | 50.6 | 39.4 | 41.1 | 56.8 | 35.0 | 48.6 | 48.8 | 36.1 | 49.0 | 34.8 | 46.1 | 43.3 |
| DAN         | 38.8 | 55.2 | 43.9 | 45.9 | 59.0 | 40.8 | 50.8 | 49.8 | 38.9 | 56.1 | 45.9 | 55.5 | 48.4 |
| DANN        | 37.9 | 54.3 | 44.4 | 41.7 | 55.6 | 36.8 | 50.7 | 50.8 | 40.1 | 55.0 | 45.0 | 54.5 | 47.2 |
| JAN         | 40.5 | 56.7 | 45.1 | 47.2 | 59.9 | 43.0 | 54.2 | 52.6 | 41.9 | 56.6 | 46.2 | 55.5 | 50.0 |
| CDAN        | 40.4 | 56.8 | 46.1 | 45.1 | 58.4 | 40.5 | 55.6 | 53.6 | 43.0 | 57.2 | 46.4 | 55.7 | 49.9 |
| MCD         | 37.5 | 52.9 | 44.0 | 44.6 | 54.5 | 41.6 | 52.0 | 51.5 | 39.7 | 55.5 | 44.6 | 52.0 | 47.5 |
| MDD         | 42.9 | 59.5 | 47.5 | 48.6 | 59.4 | 42.6 | 58.3 | 53.7 | 46.2 | 58.7 | 46.5 | 57.7 | 51.8 |
| MCC         | 37.7 | 55.7 | 42.6 | 45.4 | 59.8 | 39.9 | 54.4 | 53.1 | 37.0 | 58.1 | 46.3 | 56.2 | 48.9 |

### DomainNet accuracy on ResNet-101 (Multi-Source)

| Methods     | Origin | Avg  | :c   | :i   | :p   | :q   | :r   | :s   |
|-------------|--------|------|------|------|------|------|------|------|
| ERM         | 32.9   | 47.0 | 64.9 | 25.2 | 54.4 | 16.9 | 68.2 | 52.3 |
| MDD         | /      | 48.8 | 68.7 | 29.7 | 58.2 | 9.7  | 69.4 | 56.9 |
| Oracle      | 63.0   | 69.1 | 78.2 | 40.7 | 71.6 | 69.7 | 83.8 | 70.6 |

### Performance on ImageNet-scale dataset

|      | ResNet50, ImageNet->ImageNetR | ig_resnext101_32x8d, ImageNet->ImageSketch |
|------|-------------------------------|------------------------------------------|
| ERM  | 35.6                          | 54.9                                     |
| DAN  | 39.8                          | 55.7                                     |
| DANN | 52.7                          | 56.5                                     |
| JAN  | 41.7                          | 55.7                                     |
| CDAN | 53.9                          | 58.2                                     |
| MCD  | 46.7                          | 55.0                                     |
| AFN  | 43.0                          | 55.1                                     |
| MDD  | 56.2                          | 62.4                                     |

## Visualization

After training `DANN`, run the following command

```
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --phase analysis
```

It may take a while, then in directory ``logs/dann/Office31_A2W/visualize``, you can find
``TSNE.png``.

Following are the t-SNE of representations from ResNet50 trained on source domain and those from DANN.

<img src="./fig/resnet_A2W.png" width="300"/>
<img src="./fig/dann_A2W.png" width="300"/>

## TODO

1. Support self-training methods
2. Support translation methods
3. Add results on ViT
4. Add results on ImageNet

## Citation

If you use these methods in your research, please consider citing.

```
@inproceedings{DANN,
    author = {Ganin, Yaroslav and Lempitsky, Victor},
    Booktitle = {ICML},
    Title = {Unsupervised domain adaptation by backpropagation},
    Year = {2015}
}

@inproceedings{DAN,
    author    = {Mingsheng Long and
    Yue Cao and
    Jianmin Wang and
    Michael I. Jordan},
    title     = {Learning Transferable Features with Deep Adaptation Networks},
    booktitle = {ICML},
    year      = {2015},
}

@inproceedings{JAN,
    title={Deep transfer learning with joint adaptation networks},
    author={Long, Mingsheng and Zhu, Han and Wang, Jianmin and Jordan, Michael I},
    booktitle={ICML},
    year={2017},
}

@inproceedings{ADDA,
    title={Adversarial discriminative domain adaptation},
    author={Tzeng, Eric and Hoffman, Judy and Saenko, Kate and Darrell, Trevor},
    booktitle={CVPR},
    year={2017}
}

@inproceedings{CDAN,
    author    = {Mingsheng Long and
                Zhangjie Cao and
                Jianmin Wang and
                Michael I. Jordan},
    title     = {Conditional Adversarial Domain Adaptation},
    booktitle = {NeurIPS},
    year      = {2018}
}

@inproceedings{MCD,
    title={Maximum classifier discrepancy for unsupervised domain adaptation},
    author={Saito, Kuniaki and Watanabe, Kohei and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={CVPR},
    year={2018}
}

@InProceedings{AFN,
    author = {Xu, Ruijia and Li, Guanbin and Yang, Jihan and Lin, Liang},
    title = {Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation},
    booktitle = {ICCV},
    year = {2019}
}

@inproceedings{MDD,
    title={Bridging theory and algorithm for domain adaptation},
    author={Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael},
    booktitle={ICML},
    year={2019},
}

@inproceedings{BSP,
    title={Transferability vs. discriminability: Batch spectral penalization for adversarial domain adaptation},
    author={Chen, Xinyang and Wang, Sinan and Long, Mingsheng and Wang, Jianmin},
    booktitle={ICML},
    year={2019},
}

@inproceedings{MCC,
    author    = {Ying Jin and
                Ximei Wang and
                Mingsheng Long and
                Jianmin Wang},
    title     = {Less Confusion More Transferable: Minimum Class Confusion for Versatile
               Domain Adaptation},
    year={2020},
    booktitle={ECCV},
}

@inproceedings{FixMatch,
    title={Fixmatch: Simplifying semi-supervised learning with consistency and confidence},
    author={Sohn, Kihyuk and Berthelot, David and Carlini, Nicholas and Zhang, Zizhao and Zhang, Han and Raffel, Colin A and Cubuk, Ekin Dogus and Kurakin, Alexey and Li, Chun-Liang},
    booktitle={NIPS},
    year={2020}
}

```
