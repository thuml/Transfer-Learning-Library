# Task Adaptation for Image Classification

## Installation
Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [StanfordDogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [OxfordIIITPet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [OxfordFlowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html)
- [PatchCamelyon](https://patchcamelyon.grand-challenge.org/)
- [EuroSAT](https://github.com/phelber/eurosat)

You need to prepare following datasets manually if you want to use them:
- [Retinopathy](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
- [Resisc45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)

and prepare them following [Documentation for Retinopathy](/common/vision/datasets/retinopathy.py) and [Resisc45](/common/vision/datasets/resisc45.py).

## Supported Methods

Supported methods include:

- [Explicit inductive bias for transfer learning with convolutional networks
    (L2-SP, ICML 2018)](https://arxiv.org/abs/1802.01483)
- [Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning (BSS, NIPS 2019)](https://proceedings.neurips.cc/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf)
- [DEep Learning Transfer using Fea- ture Map with Attention for convolutional networks (DELTA, ICLR 2019)](https://openreview.net/pdf?id=rkgbwsAcYm)
- [Co-Tuning for Transfer Learning (Co-Tuning, NIPS 2020)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/co-tuning-for-transfer-learning-nips20.pdf)
- [Stochastic Normalization (StochNorm, NIPS 2020)](https://papers.nips.cc/paper/2020/file/bc573864331a9e42e4511de6f678aa83-Paper.pdf)
- [Learning Without Forgetting (LWF, ECCV 2016)](https://arxiv.org/abs/1606.09282)
- [Bi-tuning of Pre-trained Representations (Bi-Tuning)](https://arxiv.org/abs/2011.06182?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)

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
2. Convert  the format of the MoCo checkpoints to the standard format of pytorch
```shell
mkdir checkpoints
python convert_moco_to_pretrained.py checkpoints/moco_v1_200ep_pretrain.pth.tar checkpoints/moco_v1_200ep_backbone.pth checkpoints/moco_v1_200ep_fc.pth
```
3. Start training
```shell
CUDA_VISIBLE_DEVICES=0 python bi_tuning.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bi_tuning/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
```
The shell files als give the script to reproduce the [unsupervised pretrained benchmarks](/docs/talib/benchmarks/image_classification.rst#) with specified hyper-parameters.


## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{LWF,
  author    = {Zhizhong Li and
               Derek Hoiem},
  title     = {Learning without Forgetting},
  booktitle={ECCV},
  year      = {2016},
}

@inproceedings{L2SP,
  title={Explicit inductive bias for transfer learning with convolutional networks},
  author={Xuhong, LI and Grandvalet, Yves and Davoine, Franck},
  booktitle={ICML},
  year={2018},
}

@inproceedings{BSS,
  title={Catastrophic forgetting meets negative transfer: Batch spectral shrinkage for safe transfer learning},
  author={Chen, Xinyang and Wang, Sinan and Fu, Bo and Long, Mingsheng and Wang, Jianmin},
  booktitle={NeurIPS},
  year={2019}
}

@inproceedings{DELTA,
  title={Delta: Deep learning transfer using feature map with attention for convolutional networks},
  author={Li, Xingjian and Xiong, Haoyi and Wang, Hanchao and Rao, Yuxuan and Liu, Liping and Chen, Zeyu and Huan, Jun},
  booktitle={ICLR},
  year={2019}
}

@inproceedings{StocNorm,
  title={Stochastic Normalization},
  author={Kou, Zhi and You, Kaichao and Long, Mingsheng and Wang, Jianmin},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{CoTuning,
  title={Co-Tuning for Transfer Learning},
  author={You, Kaichao and Kou, Zhi and Long, Mingsheng and Wang, Jianmin},
  booktitle={NeurIPS},
  year={2020}
}

@article{BiTuning,
  title={Bi-tuning of Pre-trained Representations},
  author={Zhong, Jincheng and Wang, Ximei and Kou, Zhi and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2011.06182},
  year={2020}
}
```
