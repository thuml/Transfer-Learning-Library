# Partial Domain Adaptation for Image Classification

## Installation
It’s suggested to use **pytorch==1.7.1** and torchvision==0.8.2 in order to reproduce the benchmark results.

Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [Office31](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [VisDA2017](http://ai.bu.edu/visda-2017/)

## Supported Methods

Supported methods include:

- [Domain Adversarial Neural Network (DANN)](https://arxiv.org/abs/1505.07818)
- [Partial Adversarial Domain Adaptation (PADA)](https://arxiv.org/abs/1808.04205)
- [Importance Weighted Adversarial Nets (IWAN)](https://arxiv.org/abs/1803.09210)
- [Adaptive Feature Norm (AFN)](https://arxiv.org/pdf/1811.07456v2.pdf)

## Experiment and Results

The shell files give the script to reproduce the benchmark with specified hyper-parameters.
For example, if you want to train DANN on Office31, use the following script

```shell script
# Train a DANN on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
```

**Notations**
- ``Origin`` means the accuracy reported by the original paper.
- ``Avg`` is the accuracy reported by `TLlib`.
- ``ERM`` refers to the model trained with data from the source domain.
- ``Oracle`` refers to the model trained with data from the target domain.


We found that the accuracies of adversarial methods (including DANN) are not stable
even after the random seed is fixed, thus we repeat running adversarial methods on *Office-31* and *VisDA-2017*
for three times and report their average accuracy.

### Office-31 accuracy on ResNet-50
| Methods     | Origin | Avg  | A → W | D → W | W → D | A → D | D → A | W → A | 
|-------------|--------|------|-------|-------|-------|-------|-------|-------|
| ERM | 75.6   | 90.1 | 78.3  | 98.3  | 99.4  | 87.3  | 88.5  | 88.8  | 84.0  |
| DANN        | 43.4   | 82.4 | 60.0  | 94.9  | 98.1  | 71.3  | 84.9  | 85.0  | 
| PADA        | 92.7   | 93.8 | 86.4  | 100.0 | 100.0 | 87.3  | 93.8  | 95.4  |
| IWAN        | 94.7   | 94.8 | 91.2  | 99.7  | 99.4  | 89.8  | 94.2  | 94.3  |
| AFN         | /      | 93.1 | 87.8  | 95.6  | 99.4  | 87.9  | 93.9  | 94.1  |

### Office-Home accuracy on ResNet-50

| Methods     | Origin | Avg  | Ar → Cl | Ar → Pr | Ar → Rw | Cl → Ar | Cl → Pr | Cl → Rw | Pr → Ar | Pr → Cl | Pr → Rw | Rw → Ar | Rw → Cl | Rw → Pr |
|-------------|--------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| ERM | 53.7   | 60.1 | 42.0    | 66.9    | 78.5    | 56.4    | 55.2    | 65.4    | 57.9    | 36.0    | 75.5    | 68.7    | 43.6    | 74.8    |
| DANN        | 47.4   | 57.0 | 46.2    | 59.3    | 76.9    | 47.0    | 47.4    | 56.4    | 51.6    | 38.8    | 72.1    | 68.0    | 46.1    | 74.2    |
| PADA        | 62.1   | 65.9 | 52.9    | 69.3    | 82.8    | 59.0    | 57.5    | 66.4    | 66.0    | 41.7    | 82.5    | 78.0    | 50.2    | 84.1    |
| IWAN        | 63.6   | 71.3 | 59.2    | 76.6    | 84.0    | 67.8    | 66.7    | 69.2    | 73.3    | 55.0    | 83.9    | 79.0    | 58.3    | 82.2    |
| AFN         | 71.8   | 72.6 | 59.2    | 76.7    | 82.8    | 72.5    | 74.5    | 76.8    | 72.5    | 56.7    | 80.8    | 77.0    | 60.5    | 81.6    |

### VisDA-2017 accuracy on ResNet-50
| Methods     | Origin | Mean | plane | bcycl | bus  | car  | horse | knife | Avg  |
|-------------|--------|------|-------|-------|------|------|-------|-------|------|
| ERM | 45.3   | 50.9 | 59.2  | 31.3  | 68.7 | 73.2 | 69.3  | 3.4   | 60.0 |
| DANN        | 51.0   | 55.9 | 88.4  | 34.1  | 72.1 | 50.7 | 61.9  | 27.8  | 57.1 |
| PADA        | 53.5   | 60.5 | 89.4  | 35.1  | 72.5 | 69.2 | 86.7  | 10.1  | 66.8 |
| IWAN        | /      | 61.5 | 89.2  | 57.0  | 61.5 | 55.2 | 80.1  | 25.7  | 66.8 |
| AFN         | 67.6   | 61.0 | 79.1  | 62.7  | 73.9 | 49.6 | 79.6  | 21.0  | 64.1 |

## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{DANN,
    author = {Ganin, Yaroslav and Lempitsky, Victor},
    Booktitle = {ICML},
    Title = {Unsupervised domain adaptation by backpropagation},
    Year = {2015}
}

@InProceedings{PADA,
    author    = {Zhangjie Cao and
               Lijia Ma and
               Mingsheng Long and
               Jianmin Wang},
    title     = {Partial Adversarial Domain Adaptation},
    booktitle = {ECCV},
    year = {2018}
}

@InProceedings{IWAN,
    author    = {Jing Zhang and
               Zewei Ding and
               Wanqing Li and
               Philip Ogunbona},
    title     = {Importance Weighted Adversarial Nets for Partial Domain Adaptation},
    booktitle = {CVPR},
    year = {2018}
}

@InProceedings{AFN,
    author = {Xu, Ruijia and Li, Guanbin and Yang, Jihan and Lin, Liang},
    title = {Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation},
    booktitle = {ICCV},
    year = {2019}
}
```
