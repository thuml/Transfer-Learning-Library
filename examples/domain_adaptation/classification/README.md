# Unsupervised Domain Adaptation for Image Classification

## Installation
Example scripts can deal with [WILDS datasets](https://wilds.stanford.edu/).
You should first install ``wilds`` before using these scripts.

```
pip install wilds
```

Example scripts also support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Supported datasets include:

- Office31
- OfficeCaltech
- OfficeHome
- VisDA2017
- DomainNet
- iwildcam (WILDS)
- camelyon17 (WILDS)
- fmow (WILDS)

## Supported Methods

Supported methods include:

- Domain Adversarial Neural Network (DANN)
- Deep Adaptation Network (DAN)
- Joint Adaptation Network (JAN)
- Conditional Domain Adversarial Network (CDAN)
- Maximum Classifier Discrepancy (MCD)
- Adaptive Feature Norm (AFN)
- Margin Disparity Discrepancy (MDD)
- Minimum Class Confusion (MCC)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/unsupervised_da.rst) with specified hyper-parameters.
For example, if you want to train DANN on Office31, use the following script

```shell script
# Train a DANN on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
```

## TODO
Support methods: ADDA, BSP, AdaBN/TransNorm, CycleGAN, CyCADA
