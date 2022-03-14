# Model Selection

## Installation
Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

- [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
- [CIFAR10](http://www.cs.utoronto.ca/~kriz/cifar.html)
- [CIFAR100](http://www.cs.utoronto.ca/~kriz/cifar.html)
- [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html)
- [OxfordIIITPets](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [SUN397](https://vision.princeton.edu/projects/2010/SUN/)

## Supported Methods

Supported methods include:

- [An Information-theoretic Approach to Transferability in Task Transfer Learning (H-Score, ICIP 2019)](http://yangli-feasibility.com/home/media/icip-19.pdf)

- [LEEP: A New Measure to Evaluate Transferability of Learned Representations (LEEP, ICML 2020)](http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf)

- [Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models for Transfer Learning (LogME, ICML 2021)](https://arxiv.org/pdf/2102.11005.pdf)

- [Negative Conditional Entropy in `Transferability and Hardness of Supervised Classification Tasks (NCE, ICCV 2019)](https://arxiv.org/pdf/1908.08142v1.pdf)
    
## Experiment and Results

### Model Ranking on image classification tasks

The shell files give the scripts to ranking pre-trained models on a given dataset. For example, if you want to use LogME to calculate the transfer performance of ResNet50(ImageNet pre-trained) on Aircraft, use the following script

```shell script
# Using LogME to ranking pre-trained ResNet50 on Aircraft
# Assume you have put the datasets under the path `data/cub200`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a resnet50 -l fc --save_features
```

We use LEEP, NCE HScore and LogME to compute scores by applying 10 pre-trained models to different datasets. The correlation([Weighted kendall Tau](https://vigna.di.unimi.it/ftp/papers/WeightedTau.pdf)/Pearson Correlation) between scores and fine-tuned accuracies
are presented.

#### Model Ranking Benchmark on Aircraft

| Model        | Finetuned Acc | HScore | LEEP   | LogME | NCE    |
|--------------|---------------|--------|--------|-------|--------|
| GoogleNet    |          82.7 | 28.37  | -4.310 | 0.934 | -4.248 |
| Inception V3 |          88.8 | 43.89  | -4.202 | 0.953 | -4.170 |
| ResNet50     |          86.6 | 46.23  | -4.215 | 0.946 | -4.201 |
| ResNet101    |          85.6 | 46.13  | -4.230 | 0.948 | -4.222 |
| ResNet152    |          85.3 | 46.25  | -4.230 | 0.950 | -4.229 |
| DenseNet121  |          85.4 | 31.53  | -4.228 | 0.938 | -4.215 |
| DenseNet169  |          84.5 | 41.81  | -4.245 | 0.943 | -4.270 |
| Densenet201  |          84.6 | 46.01  | -4.206 | 0.942 | -4.189 |
| MobileNet V2 |          82.8 | 34.43  | -4.198 | 0.941 | -4.208 |
| MNasNet      |          72.8 | 35.28  | -4.192 | 0.948 | -4.195 |
| Pearson Corr |             - |  0.688 |  0.127 | 0.582 | 0.173 |
| Weighted Tau |             - |  0.664 | -0.264 | 0.595 |  0.002 |

#### Model Ranking Benchmark on Caltech101

| Model        | Finetuned Acc | HScore | LEEP   | LogME | NCE    |
|--------------|---------------|--------|--------|-------|--------|
| GoogleNet    |          91.7 | 75.88  | -1.462 | 1.228 | -0.665 |
| Inception V3 |          94.3 | 93.73  | -1.119 | 1.387 | -0.560 |
| ResNet50     |          91.8 | 91.65  | -1.020 | 1.262 | -0.616 |
| ResNet101    |          93.1 | 92.54  | -0.899 | 1.305 | -0.603 |
| ResNet152    |          93.2 | 92.91  | -0.875 | 1.324 | -0.605 |
| DenseNet121  |          91.9 | 75.02  | -0.979 | 1.172 | -0.609 |
| DenseNet169  |          92.5 | 86.37  | -0.864 | 1.212 | -0.580 |
| Densenet201  |          93.4 | 89.90  | -0.914 | 1.228 | -0.590 |
| MobileNet V2 |          89.1 | 75.82  | -1.115 | 1.150 | -0.693 |
| MNasNet      |          91.5 | 77.00  | -1.043 | 1.178 | -0.690 |
| Pearson Corr |             - |  0.748 |  0.324 | 0.794 |  0.843 |
| Weighted Tau |             - |  0.721 |  0.127 | 0.697 |  0.810 |

#### Model Ranking Benchmark on CIFAR10

| Model        | Finetuned Acc | HScore | LEEP   | LogME | NCE    |
|--------------|---------------|--------|--------|-------|--------|
| GoogleNet    |         96.2  | 5.911  | -1.385 | 0.293 | -1.139 |
| Inception V3 |         97.5  | 6.363  | -1.259 | 0.349 | -1.060 |
| ResNet50     |         96.8  | 6.567  | -1.010 | 0.388 | -1.007 |
| ResNet101    |         97.7  | 6.901  | -0.829 | 0.463 | -0.838 |
| ResNet152    |         97.9  | 6.945  | -0.838 | 0.469 | -0.851 |
| DenseNet121  |         97.2  | 6.210  | -1.035 | 0.302 | -1.006 |
| DenseNet169  |         97.4  | 6.547  | -0.934 | 0.343 | -0.946 |
| Densenet201  |         97.4  | 6.706  | -0.888 | 0.369 | -0.866 |
| MobileNet V2 |         95.7  | 5.928  | -1.100 | 0.291 | -1.089 |
| MNasNet      |         96.8  | 6.018  | -1.066 | 0.304 | -1.086 |
| Pearson Corr |             - |  0.839 |  0.604 | 0.733 |  0.786 |
| Weighted Tau |             - |  0.800 |  0.638 | 0.785 |  0.714 |

#### Model Ranking Benchmark on CIFAR100

| Model        | Finetuned Acc | HScore | LEEP   | LogME | NCE    |
|--------------|---------------|--------|--------|-------|--------|
| GoogleNet    |         83.2  | 29.33  | -3.234 | 1.037 | -2.751 |
| Inception V3 |         86.6  | 36.47  | -2.995 | 1.070 | -2.615 |
| ResNet50     |         84.5  | 40.20  | -2.612 | 1.099 | -2.516 |
| ResNet101    |         87.0  | 43.80  | -2.365 | 1.130 | -2.285 |
| ResNet152    |         87.6  | 44.19  | -2.410 | 1.133 | -2.369 |
| DenseNet121  |         84.8  | 32.13  | -2.665 | 1.029 | -2.504 |
| DenseNet169  |         85.0  | 37.51  | -2.494 | 1.051 | -2.418 |
| Densenet201  |         86.0  | 39.75  | -2.470 | 1.061 | -2.305 |
| MobileNet V2 |         80.8  | 30.36  | -2.800 | 1.039 | -2.653 |
| MNasNet      |         83.9  | 32.05  | -2.732 | 1.051 | -2.643 |
| Pearson Corr | -             | 0.815  | 0.513  | 0.698 | 0.705  |
| Weighted Tau | -             | 0.775  | 0.659  | 0.790 | 0.654  |

#### Model Ranking Benchmark on DTD

| Model        | Finetuned Acc | HScore | LEEP   | LogME | NCE   |
|--------------|---------------|--------|--------|-------|-------|
| GoogleNet    |          73.6 | 34.61  | -2.333 | 0.682 | 0.682 |
| Inception V3 |          77.2 | 57.17  | -2.135 | 0.691 | 0.691 |
| ResNet50     |          75.2 | 78.26  | -1.985 | 0.695 | 0.695 |
| ResNet101    |          76.2 | 117.23 | -1.974 | 0.689 | 0.689 |
| ResNet152    |          75.4 | 32.30  | -1.924 | 0.698 | 0.698 |
| DenseNet121  |          74.9 | 35.23  | -2.001 | 0.670 | 0.670 |
| DenseNet169  |          74.8 | 43.36  | -1.817 | 0.686 | 0.686 |
| Densenet201  |          74.5 | 45.96  | -1.926 | 0.689 | 0.689 |
| MobileNet V2 |          72.9 | 37.99  | -2.098 | 0.664 | 0.664 |
| MNasNet      |          72.8 | 38.03  | -2.033 | 0.679 | 0.679 |
| Pearson Corr | -             | 0.532  | 0.217  | 0.617 | 0.471 |
| Weighted Tau | -             | 0.416  | -0.004 | 0.550 | 0.083 |

#### Model Ranking Benchmark on OxfordIIITPets

| Model        | Finetuned Acc | HScore | LEEP   | LogME | NCE    |
|--------------|---------------|--------|--------|-------|--------|
| GoogleNet    |         91.9  | 28.02  | -1.064 | 0.854 | -0.815 |
| Inception V3 |         93.5  | 33.29  | -0.888 | 1.119 | -0.711 |
| ResNet50     |         92.5  | 32.55  | -0.805 | 0.952 | -0.721 |
| ResNet101    |         94.0  | 32.76  | -0.769 | 0.985 | -0.717 |
| ResNet152    |         94.5  | 32.86  | -0.732 | 1.009 | -0.679 |
| DenseNet121  |         92.9  | 27.09  | -0.837 | 0.797 | -0.753 |
| DenseNet169  |         93.1  | 30.09  | -0.779 | 0.829 | -0.699 |
| Densenet201  |         92.8  | 31.25  | -0.810 | 0.860 | -0.716 |
| MobileNet V2 |         90.5  | 27.83  | -0.902 | 0.765 | -0.822 |
| MNasNet      |         89.4  | 27.95  | -0.854 | 0.785 | -0.812 |
| Pearson Corr | -             | 0.427  | -0.127 | 0.589 | 0.501  |
| Weighted Tau | -             | 0.425  | -0.143 | 0.502 | 0.119  |

#### Model Ranking Benchmark on StanfordCars

| Model        | Finetuned Acc | HScore | LEEP   | LogME | NCE    |
|--------------|---------------|--------|--------|-------|--------|
| GoogleNet    |         91.0  | 41.47  | -4.612 | 1.246 | -4.312 |
| Inception V3 |         92.3  | 73.68  | -4.268 | 1.259 | -4.110 |
| ResNet50     |         91.7  | 72.94  | -4.366 | 1.253 | -4.221 |
| ResNet101    |         91.7  | 73.98  | -4.281 | 1.255 | -4.218 |
| ResNet152    |         92.0  | 76.17  | -4.215 | 1.260 | -4.142 |
| DenseNet121  |         91.5  | 45.82  | -4.437 | 1.249 | -4.271 |
| DenseNet169  |         91.5  | 63.40  | -4.286 | 1.252 | -4.175 |
| Densenet201  |         91.0  | 70.50  | -4.319 | 1.251 | -4.151 |
| MobileNet V2 |         91.0  | 51.12  | -4.463 | 1.250 | -4.306 |
| MNasNet      |         88.5  | 51.91  | -4.423 | 1.254 | -4.338 |
| Pearson Corr | -             | 0.503  | 0.433  | 0.274 | 0.695  |
| Weighted Tau | -             | 0.638  | 0.703  | 0.654 | 0.750  |

#### Model Ranking Benchmark on SUN397

| Model        | Finetuned Acc | HScore | LEEP   | LogME | NCE    |
|--------------|---------------|--------|--------|-------|--------|
| GoogleNet    |         62.0  | 71.35  | -3.744 | 1.621 | -3.055 |
| Inception V3 |         65.7  | 114.21 | -3.372 | 1.648 | -2.844 |
| ResNet50     |         64.7  | 110.39 | -3.198 | 1.638 | -2.894 |
| ResNet101    |         64.8  | 113.63 | -3.103 | 1.642 | -2.837 |
| ResNet152    |         66.0  | 116.51 | -3.056 | 1.646 | -2.822 |
| DenseNet121  |         62.3  | 72.16  | -3.311 | 1.614 | -2.945 |
| DenseNet169  |         63.0  | 95.80  | -3.165 | 1.623 | -2.903 |
| Densenet201  |         64.7  | 103.09 | -3.205 | 1.624 | -2.896 |
| MobileNet V2 |         60.5  | 75.90  | -3.338 | 1.617 | -2.968 |
| MNasNet      |         60.7  | 80.91  | -3.234 | 1.625 | -2.933 |
| Pearson Corr | -             | 0.913  | 0.428  | 0.824 | 0.782  |
| Weighted Tau | -             | 0.918  | 0.581  | 0.748 | 0.873  |

## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{bao_information-theoretic_2019,
	title = {An Information-Theoretic Approach to Transferability in Task Transfer Learning},
	booktitle = {ICIP},
	author = {Bao, Yajie and Li, Yang and Huang, Shao-Lun and Zhang, Lin and Zheng, Lizhong and Zamir, Amir and Guibas, Leonidas},
	year = {2019}
}

@inproceedings{nguyen_leep:_2020,
	title = {LEEP: A New Measure to Evaluate Transferability of Learned Representations},
	booktitle = {ICML},
	author = {Nguyen, Cuong and Hassner, Tal and Seeger, Matthias and Archambeau, Cedric},
	year = {2020}
}

@inproceedings{you_logme:_2021,
	title = {LogME: Practical Assessment of Pre-trained Models for Transfer Learning},
	booktitle = {ICML},
	author = {You, Kaichao and Liu, Yong and Wang, Jianmin and Long, Mingsheng},
	year = {2021}
}

@inproceedings{tran_transferability_2019,
	title = {Transferability and hardness of supervised classification tasks},
	booktitle = {ICCV},
	author = {Tran, Anh T. and Nguyen, Cuong V. and Hassner, Tal},
	year = {2019}
}

```