# Domain Generalization for Person Re-Identification

## Installation
Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

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

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dglib/benchmarks/re_identification.rst) with specified hyper-parameters.
For example, if you want to reproduce MixStyle on Market1501 -> DukeMTMC task, use the following script

```shell script
# Train MixStyle on Market1501 -> DukeMTMC task using ResNet 50.
# Assume you have put the datasets under the path `data/market1501` and `data/dukemtmc`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data -s Market1501 -t DukeMTMC -a resnet50 \
--mix-layers layer1 layer2 --finetune --seed 0 --log logs/mixstyle/Market2Duke
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.

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