# Domain Generalization for Person Re-Identification

## Installation

It’s suggested to use **pytorch==1.7.1** and torchvision==0.8.2 in order to reproduce the benchmark results.

Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models). You
also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [Market1501](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- [DukeMTMC](https://exposing.ai/duke_mtmc/)
- [MSMT17](https://arxiv.org/pdf/1711.08565.pdf)

## Supported Methods

Supported methods include:

- [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net (IBN-Net, 2018 ECCV)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf)
- [Domain Generalization with MixStyle (MixStyle, 2021 ICLR)](https://arxiv.org/abs/2104.02008)

## Usage

The shell files give the script to reproduce the benchmark with specified hyper-parameters. For example, if you want to
train MixStyle on Market1501 -> DukeMTMC task, use the following script

```shell script
# Train MixStyle on Market1501 -> DukeMTMC task using ResNet 50.
# Assume you have put the datasets under the path `data/market1501` and `data/dukemtmc`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data -s Market1501 -t DukeMTMC -a resnet50 \
--mix-layers layer1 layer2 --finetune --seed 0 --log logs/mixstyle/Market2Duke
```

### Experiment and Results

In our experiments, we adopt modified resnet architecture from [MMT](https://arxiv.org/pdf/2001.01526.pdf>). For a fair
comparison, we use standard cross entropy loss and triplet loss in all methods.

**Notations**

- ``Avg`` means the mAP (mean average precision) reported by `TLlib`.

### Cross dataset mAP on ResNet-50

| Methods  | Avg  | Market2Duke | Duke2Market | Market2MSMT | MSMT2Market | Duke2MSMT | MSMT2Duke |
|----------|------|-------------|-------------|-------------|-------------|-----------|-----------|
| Baseline | 23.5 | 25.6        | 29.6        | 6.3         | 31.7        | 10.1      | 37.8      |
| IBN      | 27.0 | 31.5        | 33.3        | 10.4        | 33.6        | 13.7      | 40.0      |
| MixStyle | 25.5 | 27.2        | 31.6        | 8.2         | 33.9        | 12.4      | 39.9      |

## Citation

If you use these methods in your research, please consider citing.

```
@inproceedings{IBN-Net,  
    author = {Xingang Pan, Ping Luo, Jianping Shi, and Xiaoou Tang},  
    title = {Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net},  
    booktitle = {ECCV},  
    year = {2018}  
}

@inproceedings{mixstyle,
    title={Domain Generalization with MixStyle},
    author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
    booktitle={ICLR},
    year={2021}
}
```