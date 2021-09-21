# Domain Adaptation for Person Re-Identification

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
- Similarity Preserving Generative Adversarial Network (SPGAN)
- Mutual Mean-Teaching (MMT)

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/reid_da.rst) with specified hyper-parameters.
For example, if you want to reproduce MMT on Market1501 -> DukeMTMC task, use the following script

```shell script
# Train MMT on Market1501 -> DukeMTMC task using ResNet 50.
# Assume you have put the datasets under the path `data/market1501` and `data/dukemtmc`, 
# or you are glad to download the datasets automatically from the Internet to this path

# MMT involves two training steps:
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/Market2DukeSeed0
CUDA_VISIBLE_DEVICES=0 python baseline.py data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 1 --log logs/baseline/Market2DukeSeed1

# step2: train mmt
CUDA_VISIBLE_DEVICES=0,1,2,3 python mmt.py data -t DukeMTMC -a reid_resnet50 \
--pretrained-model-1-path logs/baseline/Market2DukeSeed0/checkpoints/best.pth \
--pretrained-model-2-path logs/baseline/Market2DukeSeed1/checkpoints/best.pth \
--finetune --seed 0 --log logs/mmt/Market2Duke
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.
