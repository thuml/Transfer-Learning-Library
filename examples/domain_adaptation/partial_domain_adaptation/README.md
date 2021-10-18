# Partial Domain Adaptation for Image Classification

## Installation
Example scripts also support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [Office31](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [VisDA2017](http://ai.bu.edu/visda-2017/)

You need to prepare following datasets manually if you want to use them:
- [ImageNet](https://www.image-net.org/)

and prepare them following [Documentation for ImageNetCaltech](/common/vision/datasets/partial/imagenet_caltech.py) and [CaltechImageNet](/common/vision/datasets/partial/caltech_imagenet.py).

## Supported Methods

Supported methods include:

- [Domain Adversarial Neural Network (DANN)](https://arxiv.org/abs/1505.07818)
- [Partial Adversarial Domain Adaptation (PADA)](https://arxiv.org/abs/1808.04205)
- [Importance Weighted Adversarial Nets (IWAN)](https://arxiv.org/abs/1803.09210)
- [Adaptive Feature Norm (AFN)](https://arxiv.org/pdf/1811.07456v2.pdf)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/partial_domain_adaptation.rst) with specified hyper-parameters.
For example, if you want to train DANN on Office31, use the following script

```shell script
# Train a DANN on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.

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
