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
- [Unsupervised Data Augmentation for Consistency Training (UDA, NIPS 2020)](https://arxiv.org/pdf/1904.12848v4.pdf)
- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (FixMatch, NIPS 2020)](https://arxiv.org/abs/2001.07685)
- [Self-Tuning for Data-Efficient Deep Learning (self-tuning, ICML 2021)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/Self-Tuning-for-Data-Efficient-Deep-Learning-icml21.pdf)

## Usage

### Semi-supervised learning with supervised pre-trained model
The shell files give the script to train with supervised pre-trained model with specified hyper-parameters.
For example, if you want to train UDA on CIFAR100, use the following script

```shell script
# Semi-supervised learning on CIFAR100 (ResNet50, 400labels).
# Assume you have put the datasets under the path `data/cifar100`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/uda/cifar100_4_labels_per_class  
```
Following common practice in semi-supervised learning, we select a class-balanced subset as the labeled dataset and 
treat other samples as unlabeled data. In the above command, `num-samples-per-class` specifies how many 
labeled samples for each class. Note that the labeled subset is **deterministic with the same random seed**. Hence, if
you want to compare different algorithms with the same labeled subset, you can simply pass in the same random seed.

### Semi-supervised learning with unsupervised pre-trained model
Take MoCo as an example.

1. Download MoCo pretrained checkpoints from https://github.com/facebookresearch/moco
2. Convert  the format of the MoCo checkpoints to the standard format of pytorch
```shell
mkdir checkpoints
python convert_moco_to_pretrained.py checkpoints/moco_v2_800ep_pretrain.pth.tar checkpoints/moco_v2_800ep_backbone.pth checkpoints/moco_v2_800ep_fc.pth
```
3. Start training
```shell
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/erm_moco_pretrain/cifar100_4_labels_per_class --lr-scheduler cos -i 2000 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth
```

## Experiment and Results

We are running experiments and will release the results soon.

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
```
