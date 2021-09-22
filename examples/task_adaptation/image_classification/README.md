# Fine-tune for Image Classification

## Installation
Example scripts can deal with [tensorflow datasets](https://www.tensorflow.org/datasets).
You should first install ``tensorflow`` and ``tensorflow-datasets`` before using these scripts.

```
pip install tensorflow
pip install tensorflow-datasets
```

Example scripts also support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Supported datasets include:

- CUB200
- StanfordCars
- Aircraft
- StanfordDogs
- OxfordIIITPet
- caltech101 (tensorflow-datasets)
- cifar100 (tensorflow-datasets)
- dtd (tensorflow-datasets)
- oxford_flowers102 (tensorflow-datasets) 
- oxford_iiit_pet (tensorflow-datasets)
- sun397 (tensorflow-datasets)
- svhn_cropped (tensorflow-datasets)
- patch_camelyon (tensorflow-datasets)
- smallnorb_azimuth (tensorflow-datasets)
- smallnorb_elevation (tensorflow-datasets)


## Supported Methods

Supported methods include:

- Batch Spectral Shrinkage (BSS)
- DEep Learning Transfer using Feature Map with Attention (DELTA)
- Co-Tuning
- Stochastic Normalization (StochNorm)
- Learning Without Forgetting (LWF)
- Bi-Tuning

## Experiment and Results

### Fine-tune the supervised pre-trained model

The shell files give the script to reproduce the [supervised pretrained benchmarks](/docs/talib/benchmarks/image_classification.rst) with specified hyper-parameters.
For example, if you want to use vanilla fine-tune on CUB200, use the following script

```shell script
# Fine-tune ResNet50 on CUB200.
# Assume you have put the datasets under the path `data/cub200`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/baseline/cub200_100
```


### Fine-tune the unsupervised pre-trained model
Take MoCo as an example. 

1. Download MoCo pretrained checkpoints from https://github.com/facebookresearch/moco
2. Convert  the format of the MoCo checkpoints to to the standard format of pytorch
```shell
mkdir checkpoints
python convert_moco_to_pretrained.py checkpoints/moco_v1_200ep_pretrain.pth.tar checkpoints/moco_v1_200ep_backbone.pth checkpoints/moco_v1_200ep_fc.pth
```
3. Start training
```shell
CUDA_VISIBLE_DEVICES=0 python bi_tuning.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bi_tuning/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
```
The shell files als give the script to reproduce the [unsupervised pretrained benchmarks](/docs/talib/benchmarks/image_classification_unsupervised.rst) with specified hyper-parameters.


