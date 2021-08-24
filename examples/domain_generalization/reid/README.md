# Domain Generalization for Person Re-Identification

## Installation
Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

Following datasets can be downloaded automatically:

- Market1501
- DukeMTMC
- MSMT17

## Supported Methods

- Instance-Batch Normalization Network (IBN-Net)
- Domain Generalization with MixStyle (MixStyle)

The shell files give the script to reproduce the [benchmarks](/docs/dglib/benchmarks/reid.rst) with specified hyper-parameters.
For example, if you want to reproduce MixStyle on Market1501 -> DukeMTMC task, use the following script

```shell script
# Train MixStyle on Market1501 -> DukeMTMC task using ResNet 50.
# Assume you have put the datasets under the path `data/market1501` and `data/dukemtmc`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data -s Market1501 -t DukeMTMC -a resnet50 \
--mix-layers layer1 layer2 --finetune --seed 0 --log logs/mixstyle/Market2Duke
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.
