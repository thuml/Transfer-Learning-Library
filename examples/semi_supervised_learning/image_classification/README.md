# Semi-Supervised Learning for Image Classification

## Installation

It’s suggested to use **pytorch==1.7.1** and torchvision==0.8.2 in order to reproduce the benchmark results.

Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models). You
also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [FOOD-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [CIFAR10](http://www.cs.utoronto.ca/~kriz/cifar.html)
- [CIFAR100](http://www.cs.utoronto.ca/~kriz/cifar.html)
- [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [SUN397](https://vision.princeton.edu/projects/2010/SUN/)
- [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html)
- [OxfordIIITPets](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [OxfordFlowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)

## Supported Methods

Supported methods include:

- [Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (Pseudo Label, ICML 2013)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf)
- [Temporal Ensembling for Semi-Supervised Learning (Pi Model, ICLR 2017)](https://arxiv.org/abs/1610.02242)
- [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results (Mean Teacher, NIPS 2017)](https://arxiv.org/abs/1703.01780)
- [Self-Training With Noisy Student Improves ImageNet Classification (Noisy Student, CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf)
- [Unsupervised Data Augmentation for Consistency Training (UDA, NIPS 2020)](https://arxiv.org/pdf/1904.12848v4.pdf)
- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (FixMatch, NIPS 2020)](https://arxiv.org/abs/2001.07685)
- [Self-Tuning for Data-Efficient Deep Learning (Self-Tuning, ICML 2021)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/Self-Tuning-for-Data-Efficient-Deep-Learning-icml21.pdf)
- [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling (FlexMatch, NIPS 2021)](https://arxiv.org/abs/2110.08263)
- [Debiased Learning From Naturally Imbalanced Pseudo-Labels (DebiasMatch, CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Debiased_Learning_From_Naturally_Imbalanced_Pseudo-Labels_CVPR_2022_paper.pdf)
- [Debiased Self-Training for Semi-Supervised Learning (DST)](https://arxiv.org/abs/2202.07136)

## Usage

### Semi-supervised learning with supervised pre-trained model

The shell files give the script to train with supervised pre-trained model with specified hyper-parameters. For example,
if you want to train UDA on CIFAR100, use the following script

```shell script
# Semi-supervised learning on CIFAR100 (ResNet50, 400labels).
# Assume you have put the datasets under the path `data/cifar100`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.7 --seed 0 --log logs/uda/cifar100_4_labels_per_class
```

Following common practice in semi-supervised learning, we select a class-balanced subset as the labeled dataset and
treat other samples as unlabeled data. In the above command, `num-samples-per-class` specifies how many labeled samples
for each class. Note that the labeled subset is **deterministic with the same random seed**. Hence, if you want to
compare different algorithms with the same labeled subset, you can simply pass in the same random seed.

### Semi-supervised learning with unsupervised pre-trained model

Take MoCo as an example.

1. Download MoCo pretrained checkpoints from https://github.com/facebookresearch/moco
2. Convert the format of the MoCo checkpoints to the standard format of pytorch

```shell
mkdir checkpoints
python convert_moco_to_pretrained.py checkpoints/moco_v2_800ep_pretrain.pth.tar checkpoints/moco_v2_800ep_backbone.pth checkpoints/moco_v2_800ep_fc.pth
```

3. Start training

```shell
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --seed 0 --log logs/erm_moco_pretrain/cifar100_4_labels_per_class
```

## Experiment and Results

**Notations**

- ``Avg`` is the accuracy reported by `TLlib`.
- ``ERM`` refers to the model trained with only labeled data.
- ``Oracle`` refers to the model trained using all data as labeled data.

Below are the results of implemented methods. Other than _Oracle_, we randomly sample 4 labels per category.

### ImageNet Supervised Pre-training (ResNet-50)

| Methods      | Food101 | CIFAR10 | CIFAR100 | CUB200 | Aircraft | Cars | SUN397 | DTD  | Pets | Flowers | Caltech | Avg  |
|--------------|---------|---------|----------|--------|----------|------|--------|------|------|---------|---------|------|
| ERM          | 33.6    | 59.4    | 47.9     | 48.6   | 29.0     | 37.1 | 40.9   | 50.5 | 82.2 | 87.6    | 82.2    | 54.5 |
| Pseudo Label | 36.9    | 62.8    | 52.5     | 54.9   | 30.4     | 40.4 | 41.7   | 54.1 | 89.6 | 93.5    | 85.1    | 58.4 |
| Pi Model     | 34.2    | 66.9    | 48.5     | 47.9   | 26.7     | 37.4 | 40.9   | 51.9 | 83.5 | 92.0    | 82.2    | 55.6 |
| Mean Teacher | 40.4    | 78.1    | 58.5     | 52.8   | 32.0     | 45.6 | 40.2   | 53.8 | 86.8 | 92.8    | 83.7    | 60.4 |
| UDA          | 41.9    | 73.0    | 59.8     | 55.4   | 33.5     | 42.7 | 42.1   | 49.7 | 88.0 | 93.4    | 85.3    | 60.4 |
| FixMatch     | 36.2    | 74.5    | 58.0     | 52.6   | 27.1     | 44.8 | 40.8   | 50.2 | 87.8 | 93.6    | 83.2    | 59.0 |
| Self Tuning  | 41.4    | 70.9    | 57.2     | 60.5   | 37.0     | 59.8 | 43.5   | 51.7 | 88.4 | 93.5    | 89.1    | 63.0 |
| FlexMatch    | 48.1    | 94.2    | 69.2     | 65.1   | 38.0     | 55.3 | 50.2   | 55.6 | 91.5 | 94.6    | 89.4    | 68.3 |
| DebiasMatch  | 57.1    | 92.4    | 69.0     | 66.2   | 41.5     | 65.4 | 48.3   | 54.2 | 90.2 | 95.4    | 89.3    | 69.9 |
| DST          | 58.1    | 93.5    | 67.8     | 68.6   | 44.9     | 68.6 | 47.0   | 56.3 | 91.5 | 95.1    | 90.3    | 71.1 |
| Oracle       | 85.5    | 97.5    | 86.3     | 81.1   | 85.1     | 91.1 | 64.1   | 68.8 | 93.2 | 98.1    | 92.6    | 85.8 |

### ImageNet Unsupervised Pre-training (ResNet-50, MoCo v2)

| Methods      | Food101 | CIFAR10 | CIFAR100 | CUB200 | Aircraft | Cars | SUN397 | DTD  | Pets | Flowers | Caltech | Avg  |
|--------------|---------|---------|----------|--------|----------|------|--------|------|------|---------|---------|------|
| ERM          | 33.5    | 63.0    | 50.8     | 39.4   | 28.1     | 40.3 | 40.7   | 53.7 | 65.4 | 87.5    | 82.8    | 53.2 |
| Pseudo Label | 33.6    | 71.9    | 53.8     | 42.7   | 30.9     | 51.2 | 41.2   | 55.2 | 69.3 | 94.2    | 86.2    | 57.3 |
| Pi Model     | 32.7    | 77.9    | 50.9     | 33.6   | 27.2     | 34.4 | 41.1   | 54.9 | 66.7 | 91.4    | 84.1    | 54.1 |
| Mean Teacher | 36.8    | 79.0    | 56.7     | 43.0   | 33.0     | 53.9 | 39.5   | 54.5 | 67.8 | 92.7    | 83.3    | 58.2 |
| UDA          | 39.5    | 91.3    | 60.0     | 41.9   | 36.2     | 39.7 | 41.7   | 51.5 | 71.0 | 93.7    | 86.5    | 59.4 |
| FixMatch     | 44.3    | 86.1    | 58.0     | 42.7   | 38.0     | 55.4 | 42.4   | 53.1 | 67.9 | 95.2    | 83.4    | 60.6 |
| Self Tuning  | 34.0    | 63.6    | 51.7     | 43.3   | 32.2     | 50.2 | 40.7   | 52.7 | 68.2 | 91.8    | 87.7    | 56.0 |
| FlexMatch    | 50.2    | 96.6    | 69.2     | 49.4   | 41.3     | 62.5 | 47.2   | 54.5 | 72.4 | 94.8    | 89.4    | 66.1 |
| DebiasMatch  | 54.2    | 95.5    | 68.1     | 49.1   | 40.9     | 73.0 | 47.6   | 54.4 | 76.6 | 95.5    | 88.7    | 67.6 |
| DST          | 57.1    | 95.0    | 68.2     | 53.6   | 47.7     | 72.0 | 46.8   | 56.0 | 76.3 | 95.6    | 90.1    | 68.9 |
| Oracle       | 87.0    | 98.2    | 87.9     | 80.6   | 88.7     | 92.7 | 63.9   | 73.8 | 90.6 | 97.8    | 93.1    | 86.8 |

## TODO

1. support multi-gpu training
2. add training from scratch code and results

## Citation

If you use these methods in your research, please consider citing.

```
@inproceedings{pseudo_label,
    title={Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks},
    author={Lee, Dong-Hyun and others},
    booktitle={ICML},
    year={2013}
}

@inproceedings{pi_model,
    title={Temporal ensembling for semi-supervised learning},
    author={Laine, Samuli and Aila, Timo},
    booktitle={ICLR},
    year={2017}
}

@inproceedings{mean_teacher,
    title={Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results},
    author={Tarvainen, Antti and Valpola, Harri},
    booktitle={NIPS},
    year={2017}
}

@inproceedings{noisy_student,
    title={Self-training with noisy student improves imagenet classification},
    author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V},
    booktitle={CVPR},
    year={2020}
}

@inproceedings{UDA,
    title={Unsupervised data augmentation for consistency training},
    author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Thang and Le, Quoc},
    booktitle={NIPS},
    year={2020}
}

@inproceedings{FixMatch,
    title={Fixmatch: Simplifying semi-supervised learning with consistency and confidence},
    author={Sohn, Kihyuk and Berthelot, David and Carlini, Nicholas and Zhang, Zizhao and Zhang, Han and Raffel, Colin A and Cubuk, Ekin Dogus and Kurakin, Alexey and Li, Chun-Liang},
    booktitle={NIPS},
    year={2020}
}

@inproceedings{SelfTuning,
    title={Self-tuning for data-efficient deep learning},
    author={Wang, Ximei and Gao, Jinghan and Long, Mingsheng and Wang, Jianmin},
    booktitle={ICML},
    year={2021}
}

@inproceedings{FlexMatch,
    title={Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling},
    author={Zhang, Bowen and Wang, Yidong and Hou, Wenxin and Wu, Hao and Wang, Jindong and Okumura, Manabu and Shinozaki, Takahiro},
    booktitle={NeurIPS},
    year={2021}
}

@inproceedings{DebiasMatch,
    title={Debiased Learning from Naturally Imbalanced Pseudo-Labels},
    author={Wang, Xudong and Wu, Zhirong and Lian, Long and Yu, Stella X},
    booktitle={CVPR},
    year={2022}
}

@article{DST,
    title={Debiased Self-Training for Semi-Supervised Learning},
    author={Chen, Baixu and Jiang, Junguang and Wang, Ximei and Wang, Jianmin and Long, Mingsheng},
    journal={arXiv preprint arXiv:2202.07136},
    year={2022}
}
```
