# Task Adaptation for Image Classification

## Installation

Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models). You
need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [StanfordDogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [OxfordIIITPets](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [OxfordFlowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html)
- [PatchCamelyon](https://patchcamelyon.grand-challenge.org/)
- [EuroSAT](https://github.com/phelber/eurosat)

You need to prepare following datasets manually if you want to use them:

- [Retinopathy](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
- [Resisc45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)

and prepare them following [Documentation for Retinopathy](/common/vision/datasets/retinopathy.py)
and [Resisc45](/common/vision/datasets/resisc45.py).

## Supported Methods

Supported methods include:

- [Explicit inductive bias for transfer learning with convolutional networks
  (L2-SP, ICML 2018)](https://arxiv.org/abs/1802.01483)
- [Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning (BSS, NIPS 2019)](https://proceedings.neurips.cc/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf)
- [DEep Learning Transfer using Fea- ture Map with Attention for convolutional networks (DELTA, ICLR 2019)](https://openreview.net/pdf?id=rkgbwsAcYm)
- [Co-Tuning for Transfer Learning (Co-Tuning, NIPS 2020)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/co-tuning-for-transfer-learning-nips20.pdf)
- [Stochastic Normalization (StochNorm, NIPS 2020)](https://papers.nips.cc/paper/2020/file/bc573864331a9e42e4511de6f678aa83-Paper.pdf)
- [Learning Without Forgetting (LWF, ECCV 2016)](https://arxiv.org/abs/1606.09282)
- [Bi-tuning of Pre-trained Representations (Bi-Tuning)](https://arxiv.org/abs/2011.06182?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)

## Experiment and Results

We follow the common practice in the community as described
in [Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning (BSS, NIPS 2019)](https://proceedings.neurips.cc/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf)
.

Training iterations and data augmentations are kept the same for different task-adaptation methods for a fair
comparison.

Hyper-parameters of each method are selected by the performance on target validation data.

### Fine-tune the supervised pre-trained model

The shell files give the script to reproduce the supervised pretrained benchmarks with specified hyper-parameters. For
example, if you want to use vanilla fine-tune on CUB200, use the following script

```shell script
# Fine-tune ResNet50 on CUB200.
# Assume you have put the datasets under the path `data/cub200`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/erm/cub200_100
```

#### Vision Benchmark on ResNet-50

|                 | Food101 | CIFAR10 | CIFAR100 | SUN397 | Standford Cars | FGVC Aircraft  | DTD   | Oxford-IIIT Pets | Caltech-101    | Oxford 102 Flowers | average |
|-----------------|---------|---------|----------|--------|----------------|----------------|-------|------------------|----------------|--------------------|---------|
| Accuracy metric | top1    | top1    | top1     | top1   | top1           | mean per-class | top-1 | mean per-class   | mean per-class | mean per-class     |         |
| Baseline        | 85.1    | 96.6    | 84.1     | 63.7   | 87.8           | 80.1           | 70.8  | 93.2             | 91.1           | 93.0               | 84.6    |
| LWF             | 83.9    | 96.5    | 83.6     | 64.1   | 87.4           | 82.2           | 72.2  | 94.0             | 89.8           | 92.9               | 84.7    |
| DELTA           | 83.8    | 95.9    | 83.7     | 64.5   | 88.1           | 82.3           | 72.2  | 94.2             | 90.1           | 93.1               | 84.8    |
| BSS             | 85.0    | 96.6    | 84.2     | 63.5   | 88.4           | 81.8           | 70.2  | 93.3             | 91.6           | 92.7               | 84.7    |
| StochNorm       | 85.0    | 96.8    | 83.9     | 63.0   | 87.7           | 81.5           | 71.3  | 93.6             | 90.5           | 92.9               | 84.6    |
| Bi-Tuning       | 85.7    | 97.1    | 84.3     | 64.2   | 90.3           | 84.8           | 70.6  | 93.5             | 91.5           | 94.5               | 85.7    |

#### CUB-200-2011 on ResNet-50 (Supervised Pre-trained)

| CUB200    | 15%  | 30%  | 50%  | 100% | Avg  |
|-----------|------|------|------|------|------|
| ERM  | 51.2 | 64.6 | 74.6 | 81.8 | 68.1 |
| lwf       | 56.7 | 66.8 | 73.4 | 81.5 | 69.6 |
| BSS       | 53.4 | 66.7 | 76.0 | 82.0 | 69.5 |
| delta     | 54.8 | 67.3 | 76.3 | 82.3 | 70.2 |
| StochNorm | 54.8 | 66.8 | 75.8 | 82.2 | 69.9 |
| Co-tuning | 57.6 | 70.1 | 77.3 | 82.5 | 71.9 |
| bi-tuning | 55.8 | 69.3 | 77.2 | 83.1 | 71.4 |

#### Stanford Cars on ResNet-50 (Supervised Pre-trained)

| Standford Cars | 15%  | 30%  | 50%  | 100% | Avg  |
|----------------|------|------|------|------|------|
| ERM       | 41.1 | 65.9 | 78.4 | 87.8 | 68.3 |
| lwf            | 44.9 | 67.0 | 77.6 | 87.5 | 69.3 |
| BSS            | 43.3 | 67.6 | 79.6 | 88.0 | 69.6 |
| delta          | 45.0 | 68.4 | 79.6 | 88.4 | 70.4 |
| StochNorm      | 44.4 | 68.1 | 79.3 | 87.9 | 69.9 |
| Co-tuning      | 49.0 | 70.6 | 81.9 | 89.1 | 72.7 |
| bi-tuning      | 48.3 | 72.8 | 83.3 | 90.2 | 73.7 |

#### FGVC Aircraft on ResNet-50 (Supervised Pre-trained)

| FGVC Aircraft | 15%  | 30%  | 50%  | 100% | Avg  |
|---------------|------|------|------|------|------|
| ERM      | 41.6 | 57.8 | 68.7 | 80.2 | 62.1 |
| lwf           | 44.1 | 60.6 | 68.7 | 82.4 | 64.0 |
| BSS           | 43.6 | 59.5 | 69.6 | 81.2 | 63.5 |
| delta         | 44.4 | 61.9 | 71.4 | 82.7 | 65.1 |
| StochNorm     | 44.3 | 60.6 | 70.1 | 81.5 | 64.1 |
| Co-tuning     | 45.9 | 61.2 | 71.3 | 82.2 | 65.2 |
| bi-tuning     | 47.2 | 64.3 | 73.7 | 84.3 | 67.4 |

### Fine-tune the unsupervised pre-trained model

Take MoCo as an example.

1. Download MoCo pretrained checkpoints from https://github.com/facebookresearch/moco
2. Convert the format of the MoCo checkpoints to the standard format of pytorch

```shell
mkdir checkpoints
python convert_moco_to_pretrained.py checkpoints/moco_v1_200ep_pretrain.pth.tar checkpoints/moco_v1_200ep_backbone.pth checkpoints/moco_v1_200ep_fc.pth
```

3. Start training

```shell
CUDA_VISIBLE_DEVICES=0 python bi_tuning.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bi_tuning/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
```

#### CUB-200-2011 on ResNet-50 (MoCo Pre-trained)

| CUB200    | 15%  | 30%  | 50%  | 100% | Avg  |
|-----------|------|------|------|------|------|
| ERM  | 28.0 | 48.2 | 62.7 | 75.6 | 53.6 |
| lwf       | 28.8 | 50.1 | 62.8 | 76.2 | 54.5 |
| BSS       | 30.9 | 50.3 | 63.7 | 75.8 | 55.2 |
| delta     | 27.9 | 51.4 | 65.9 | 74.6 | 55.0 |
| StochNorm | 20.8 | 44.9 | 60.1 | 72.8 | 49.7 |
| Co-tuning | 29.1 | 50.1 | 63.8 | 75.9 | 54.7 |
| bi-tuning | 32.4 | 51.8 | 65.7 | 76.1 | 56.5 |

#### Stanford Cars on ResNet-50 (MoCo Pre-trained)

| Standford Cars | 15%  | 30%  | 50%  | 100% | Avg  |
|----------------|------|------|------|------|------|
| ERM       | 42.5 | 71.2 | 83.0 | 90.1 | 71.7 |
| lwf            | 44.2 | 71.7 | 82.9 | 90.5 | 72.3 |
| BSS            | 45.0 | 71.5 | 83.8 | 90.1 | 72.6 |
| delta          | 45.9 | 72.9 | 82.5 | 88.9 | 72.6 |
| StochNorm      | 40.3 | 66.2 | 78.0 | 86.2 | 67.7 |
| Co-tuning      | 44.2 | 72.6 | 83.3 | 90.3 | 72.6 |
| bi-tuning      | 45.6 | 72.8 | 83.2 | 90.8 | 73.1 |

#### FGVC Aircraft on ResNet-50 (MoCo Pre-trained)

| FGVC Aircraft | 15%  | 30%  | 50%  | 100% | Avg  |
|---------------|------|------|------|------|------|
| ERM      | 45.8 | 67.6 | 78.8 | 88.0 | 70.1 |
| lwf           | 48.5 | 68.5 | 78.0 | 87.9 | 70.7 |
| BSS           | 47.7 | 69.1 | 79.2 | 88.0 | 71.0 |
| delta         | -    | -    | -    | -    | -    |
| StochNorm     | 45.4 | 68.8 | 76.7 | 86.1 | 69.3 |
| Co-tuning     | 48.2 | 68.5 | 78.7 | 87.3 | 70.7 |
| bi-tuning     | 46.4 | 69.6 | 79.4 | 87.9 | 70.8 |

## Citation

If you use these methods in your research, please consider citing.

```
@inproceedings{LWF,
    author    = {Zhizhong Li and
                Derek Hoiem},
    title     = {Learning without Forgetting},
    booktitle={ECCV},
    year      = {2016},
}

@inproceedings{L2SP,
    title={Explicit inductive bias for transfer learning with convolutional networks},
    author={Xuhong, LI and Grandvalet, Yves and Davoine, Franck},
    booktitle={ICML},
    year={2018},
}

@inproceedings{BSS,
    title={Catastrophic forgetting meets negative transfer: Batch spectral shrinkage for safe transfer learning},
    author={Chen, Xinyang and Wang, Sinan and Fu, Bo and Long, Mingsheng and Wang, Jianmin},
    booktitle={NeurIPS},
    year={2019}
}

@inproceedings{DELTA,
    title={Delta: Deep learning transfer using feature map with attention for convolutional networks},
    author={Li, Xingjian and Xiong, Haoyi and Wang, Hanchao and Rao, Yuxuan and Liu, Liping and Chen, Zeyu and Huan, Jun},
    booktitle={ICLR},
    year={2019}
}

@inproceedings{StocNorm,
    title={Stochastic Normalization},
    author={Kou, Zhi and You, Kaichao and Long, Mingsheng and Wang, Jianmin},
    booktitle={NeurIPS},
    year={2020}
}

@inproceedings{CoTuning,
    title={Co-Tuning for Transfer Learning},
    author={You, Kaichao and Kou, Zhi and Long, Mingsheng and Wang, Jianmin},
    booktitle={NeurIPS},
    year={2020}
}

@article{BiTuning,
    title={Bi-tuning of Pre-trained Representations},
    author={Zhong, Jincheng and Wang, Ximei and Kou, Zhi and Wang, Jianmin and Long, Mingsheng},
    journal={arXiv preprint arXiv:2011.06182},
    year={2020}
}
```
