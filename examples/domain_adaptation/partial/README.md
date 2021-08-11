# Partial Domain Adaptation for Image Classification

## Installation
Example scripts also support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- Office31
- OfficeCaltech
- OfficeHome
- VisDA2017
- DomainNet

You need to prepare following datasets manually if you want to use them:
- ImageNet

and prepare them following ``common.vision.datasets.partial.imagenet_caltech.py`` and ``common.vision.datasets.partial.caltech_imagenet.py``.

## Supported Methods

Supported methods include:

- Domain Adversarial Neural Network (DANN)
- Partial Adversarial Domain Adaptation (PADA)
- Importance Weighted Adversarial Nets (IWAN)
- Adaptive Feature Norm (AFN)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/partial_da.rst) with specified hyper-parameters.
For example, if you want to train DANN on Office31, use the following script

```shell script
# Train a DANN on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.
