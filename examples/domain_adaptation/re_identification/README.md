# Domain Adaptation for Person Re-Identification

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
- [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification (MMT, 2020 ICLR)](https://arxiv.org/abs/2001.01526)
- [Similarity Preserving Generative Adversarial Network (SPGAN, 2018 CVPR)](https://arxiv.org/pdf/1811.10551.pdf)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/re_identification.rst) with specified hyper-parameters.
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

## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{IBN-Net,  
    author = {Xingang Pan, Ping Luo, Jianping Shi, and Xiaoou Tang},  
    title = {Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net},  
    booktitle = {ECCV},  
    year = {2018}  
}

@inproceedings{SPGAN,
    title={Image-image domain adaptation with preserved self-similarity and domain-dissimilarity for person re-identification},
    author={Deng, Weijian and Zheng, Liang and Ye, Qixiang and Kang, Guoliang and Yang, Yi and Jiao, Jianbin},
    booktitle={CVPR},
    year={2018}
}

@inproceedings{
    MMT,
    title={Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification},
    author={Yixiao Ge and Dapeng Chen and Hongsheng Li},
    booktitle={ICLR},
    year={2020},
}
```