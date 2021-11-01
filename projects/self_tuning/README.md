# Self-Tuning

In this repository, we implement self-tuning and various SSL (semi-supervised learning) algorithms in Trans-Learn.

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

## Supported Methods

Supported methods include:

- Pseudo Label (directly utilize model predictions as pseudo labels on unlabeled samples)
- [Temporal Ensembling for Semi-Supervised Learning (pi model, ICLR 2017)](https://arxiv.org/abs/1610.02242)
- [Weight-averaged consistency targets improve semi-supervised deep learning results (mean teacher, NIPS 2017)](https://openreview.net/references/pdf?id=ry8u21rtl)
- [Unsupervised Data Augmentation for Consistency Training (uda, NIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/44feb0096faa8326192570788b38c1d1-Paper.pdf)
- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (FixMatch, NIPS 2020)](https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf)
- [Self-Tuning for Data-Efficient Deep Learning (self-tuning, ICML 2021)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/Self-Tuning-for-Data-Efficient-Deep-Learning-icml21.pdf)

## Experiments and Results

### SSL with supervised pre-trained model

The shell files give the script to reproduce our [results](benchmark.md) with specified hyper-parameters. For example,
if you want to run baseline on CUB200 with 15% labeled samples, use the following script

```shell script
# SSL with ResNet50 backbone on CUB200.
# Assume you have put the datasets under the path `data/cub200`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/baseline/cub200_15
```

### SSL with unsupervised pre-trained model

Take MoCo as an example.

1. Download MoCo pretrained checkpoints from https://github.com/facebookresearch/moco
2. Convert the format of the MoCo checkpoints to the standard format of pytorch

```shell
mkdir checkpoints
python convert_moco_to_pretrained.py checkpoints/moco_v1_200ep_pretrain.pth.tar checkpoints/moco_v1_200ep_backbone.pth checkpoints/moco_v1_200ep_fc.pth
```

3. Start training

```shell
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/baseline_moco/cub200_15 \
  --pretrained checkpoints/moco_v1_200ep_backbone.pth
```

## TODO

Support datasets: CIFAR10, ImageNet

## Citation

If you use these methods in your research, please consider citing.

```
@inproceedings{pi-model,
    title={Temporal ensembling for semi-supervised learning},
    author={Laine, Samuli and Aila, Timo},
    booktitle={ICLR},
    year={2017}
}
@inproceedings{mean-teacher,
    title={Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results},
    author={Tarvainen, Antti and Valpola, Harri},
    booktitle={NIPS},
    year={2017}
}
@inproceedings{uda,
    title={Unsupervised data augmentation for consistency training},
    author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V},
    booktitle={NIPS},
    year={2019}
}
@inproceedings{fixmatch,
    title={Fixmatch: Simplifying semi-supervised learning with consistency and confidence},
    author={Sohn, Kihyuk and Berthelot, David and Li, Chun-Liang and Zhang, Zizhao and Carlini, Nicholas and Cubuk, Ekin D and Kurakin, Alex and Zhang, Han and Raffel, Colin},
    booktitle={NIPS},
    year={2020}
}
@inproceedings{self-tuning,
    title={Self-tuning for data-efficient deep learning},
    author={Wang, Ximei and Gao, Jinghan and Long, Mingsheng and Wang, Jianmin},
    booktitle={ICML},
    year={2021},
}
```
