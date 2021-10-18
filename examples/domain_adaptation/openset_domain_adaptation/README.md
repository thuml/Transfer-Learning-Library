# Open-set Domain Adaptation for Image Classification

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

## Supported Methods

Supported methods include:

- [Open Set Domain Adaptation (OSBP)](https://arxiv.org/abs/1804.10427)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/openset_domain_adaptation.rst) with specified hyper-parameters.
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
@InProceedings{OSBP,
    author = {Saito, Kuniaki and Yamamoto, Shohei and Ushiku, Yoshitaka and Harada, Tatsuya},
    title = {Open Set Domain Adaptation by Backpropagation},
    booktitle = {ECCV},
    year = {2018}
}
```
