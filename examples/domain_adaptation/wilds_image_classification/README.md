# Unsupervised Domain Adaptation for WILDS (Image Classification)

## Installation

It’s suggested to use **pytorch==1.9.0** in order to reproduce the benchmark results.

You need to install **apex** following ``https://github.com/NVIDIA/apex``. Then run

```
pip install -r requirements.txt
```

## Dataset

Following datasets can be downloaded automatically:

- [DomainNet](http://ai.bu.edu/M3SDA/)
- [iwildcam (WILDS)](https://wilds.stanford.edu/datasets/)
- [camelyon17 (WILDS)](https://wilds.stanford.edu/datasets/)
- [fmow (WILDS)](https://wilds.stanford.edu/datasets/)

## Supported Methods

Supported methods include:

- [Domain Adversarial Neural Network (DANN)](https://arxiv.org/abs/1505.07818)
- [Deep Adaptation Network (DAN)](https://arxiv.org/pdf/1502.02791)
- [Joint Adaptation Network (JAN)](https://arxiv.org/abs/1605.06636)
- [Conditional Domain Adversarial Network (CDAN)](https://arxiv.org/abs/1705.10667)
- [Margin Disparity Discrepancy (MDD)](https://arxiv.org/abs/1904.05801)
- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (FixMatch)](https://arxiv.org/abs/2001.07685)

## Usage

Our code is based
on [https://github.com/NVIDIA/apex/edit/master/examples/imagenet](https://github.com/NVIDIA/apex/edit/master/examples/imagenet)
. It implements Automatic Mixed Precision (Amp) training of popular model architectures, such as ResNet, AlexNet, and
VGG, on the WILDS dataset.  
Command-line flags forwarded to `amp.initialize` are used to easily manipulate and switch between various pure and mixed
precision "optimization levels" or `opt_level`s.  
For a detailed explanation of `opt_level`s, see the [updated API guide](https://nvidia.github.io/apex/amp.html).

The shell files give all the training scripts we use, e.g.,

```
CUDA_VISIBLE_DEVICES=0 python erm.py data/wilds -d "fmow" --aa "v0" --arch "densenet121" \
  --lr 0.1 --opt-level O1 --deterministic --vflip 0.5 --log logs/erm/fmow/lr_0_1_aa_v0_densenet121
```

## Results

### Performance on WILDS-FMoW (DenseNet-121)

| Methods | Val Avg Acc | Test Avg Acc | Val Worst-region Acc | Test Worst-region Acc |
|---------|-------------|--------------|----------------------|-----------------------|
| ERM     | 59.8        | 53.3         | 50.2                 | 32.2                  |
| DANN    | 60.6        | 54.2         | 49.1                 | 34.8                  |
| DAN     | 61.7        | 55.5         | 48.3                 | 35.3                  |
| JAN     | 61.5        | 55.3         | 50.6                 | 36.3                  |
| CDAN    | 60.7        | 55.0         | 47.4                 | 35.5                  |
| MDD     | 60.1        | 55.1         | 49.3                 | 35.9                  |
| FixMatch| 61.1        | 55.1         | 51.8                 | 37.4                  |

### Performance on WILDS-IWildCAM (ResNet50)

| Methods | Val Avg Acc | Test Avg Acc | Val F1 macro | Test F1 macro |
|---------|-------------|--------------|--------------|---------------|
| ERM     | 59.9        | 72.6         | 36.3         | 32.9          |
| DANN    | 57.4        | 70.1         | 35.8         | 32.2          |
| DAN     | 63.7        | 69.4         | 39.1         | 31.6          |
| JAN     | 62.4        | 68.7         | 37.6         | 31.5          |
| CDAN    | 57.6        | 71.2         | 37.0         | 30.6          |
| MDD     | 58.3        | 73.5         | 35.0         | 30.0          |

### Visualization

We use tensorboard to record the training process and visualize the outputs of the models.

```
tensorboard --logdir=logs
```

### Distributed training

We uses `apex.parallel.DistributedDataParallel` (DDP) for multiprocess training with one GPU per process.

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666  erm.py data/wilds -d "fmow" --aa "v0" --arch "densenet121" \
  --lr 0.1 --opt-level O1 --deterministic --vflip 0.5 -j 8 --log logs/erm/fmow/lr_0_1_aa_v0_densenet121_bs_128
```

## TODO

1. update experiment results
2. support DomainNet
3. support camelyon17
4. support self-training methods
5. support self-supervised methods

## Citation

If you use these methods in your research, please consider citing.

```
@inproceedings{DANN,
    author = {Ganin, Yaroslav and Lempitsky, Victor},
    Booktitle = {ICML},
    Title = {Unsupervised domain adaptation by backpropagation},
    Year = {2015}
}

@inproceedings{DAN,
    author    = {Mingsheng Long and
    Yue Cao and
    Jianmin Wang and
    Michael I. Jordan},
    title     = {Learning Transferable Features with Deep Adaptation Networks},
    booktitle = {ICML},
    year      = {2015},
}

@inproceedings{JAN,
    title={Deep transfer learning with joint adaptation networks},
    author={Long, Mingsheng and Zhu, Han and Wang, Jianmin and Jordan, Michael I},
    booktitle={ICML},
    year={2017},
}

@inproceedings{CDAN,
    author    = {Mingsheng Long and
                Zhangjie Cao and
                Jianmin Wang and
                Michael I. Jordan},
    title     = {Conditional Adversarial Domain Adaptation},
    booktitle = {NeurIPS},
    year      = {2018}
}

@inproceedings{MDD,
    title={Bridging theory and algorithm for domain adaptation},
    author={Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael},
    booktitle={ICML},
    year={2019},
}

@inproceedings{FixMatch,
    title={Fixmatch: Simplifying semi-supervised learning with consistency and confidence},
    author={Sohn, Kihyuk and Berthelot, David and Carlini, Nicholas and Zhang, Zizhao and Zhang, Han and Raffel, Colin A and Cubuk, Ekin Dogus and Kurakin, Alexey and Li, Chun-Liang},
    booktitle={NIPS},
    year={2020}
}

```
