# Unsupervised Domain Adaptation for Person Re-Identification

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
- [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification (MMT, 2020 ICLR)](https://arxiv.org/abs/2001.01526)
- [Similarity Preserving Generative Adversarial Network (SPGAN, 2018 CVPR)](https://arxiv.org/pdf/1811.10551.pdf)

## Usage

The shell files give the script to reproduce the benchmark with specified hyper-parameters. For example, if you want to
train MMT on Market1501 -> DukeMTMC task, use the following script

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

### Experiment and Results
In our experiments, we adopt modified resnet architecture from [MMT](https://arxiv.org/pdf/2001.01526.pdf>). For a fair comparison,
we use standard cross entropy loss and triplet loss in all methods. For methods that utilize clustering algorithms, 
we adopt kmeans or DBSCAN and report both results.

**Notations**
- ``Avg`` means the mAP (mean average precision) reported by `TLlib`.
- ``Baseline_Cluster`` represents the strong baseline in [MMT](https://arxiv.org/pdf/2001.01526.pdf>).

### Cross dataset mAP on ResNet-50

| Methods                  | Avg  | Market2Duke | Duke2Market | Market2MSMT | MSMT2Market | Duke2MSMT | MSMT2Duke |
|--------------------------|------|-------------|-------------|-------------|-------------|-----------|-----------|
| Baseline                 | 27.1 | 32.4        | 31.4        | 8.2         | 36.7        | 11.0      | 43.1      |
| IBN                      | 30.0 | 35.2        | 36.5        | 11.3        | 38.7        | 14.1      | 44.3      |
| SPGAN                    | 30.7 | 34.4        | 35.4        | 14.1        | 40.2        | 16.1      | 43.8      |
| Baseline_Cluster(kmeans) | 45.1 | 52.8        | 59.5        | 19.0        | 62.6        | 20.3      | 56.2      |
| Baseline_Cluster(dbscan) | 54.9 | 62.5        | 73.5        | 25.2        | 77.9        | 25.3      | 65.0      |
| MMT(kmeans)              | 55.4 | 63.7        | 72.5        | 26.2        | 75.8        | 28.0      | 66.1      |
| MMT(dbscan)              | 60.0 | 68.2        | 80.0        | 28.2        | 82.5        | 31.2      | 70.0      |

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

@inproceedings{MMT,
    title={Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification},
    author={Yixiao Ge and Dapeng Chen and Hongsheng Li},
    booktitle={ICLR},
    year={2020},
}
```