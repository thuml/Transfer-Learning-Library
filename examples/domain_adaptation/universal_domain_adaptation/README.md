# Universal Domain Adaptation for Image Classification

## Installation

It’s suggested to use **pytorch==1.10.0** and torchvision==0.11.0 in order to reproduce the benchmark results. Example
scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models). You also need
to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [Office31](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [VisDA2017](http://ai.bu.edu/visda-2017/)
- [DomainNet](http://ai.bu.edu/M3SDA/)

## Supported Methods

- [Calibrated Multiple Uncertainties (CMU)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/learning-to-detect-open-classes-for-universal-domain-adaptation-eccv20.pdf)

## Experiment and Results

The shell files give the script to reproduce the benchmark with specified hyper-parameters. For example, if you want to
train CMU on Office31, use the following script

```shell script
# Train a CMU on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s A -t W -a resnet50 \
  --lr 0.001 --threshold 0.7 --src-threshold 0.4 --cut 0.1 --seed 0 --log logs/cmu/Office31_A2W
```

**Notations**
- ``Avg`` is the accuracy reported by `TLlib`.
- ``ERM`` refers to the model trained with data from the source domain.

### Office-31 accuracy on ResNet-50

| Methods | Avg  | A → W | D → W | W → D | A → D | D → A | W → A |
|---------|------|-------|-------|-------|-------|-------|-------|
| ERM     | 77.8 | 74.5  | 87.4  | 87.8  | 73.4  | 72.7  | 70.7  |
| CMU     | 78.2 | 76.1  | 89.3  | 92.6  | 77.4  | 68.1  | 65.9  |

### Office-Home accuracy on ResNet-50

| Methods | Avg  | Ar → Cl | Ar → Pr | Ar → Rw | Cl → Ar | Cl → Pr | Cl → Rw | Pr → Ar | Pr → Cl | Pr → Rw | Rw → Ar | Rw → Cl | Rw → Pr |
|---------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| ERM     | 66.2 | 55.0    | 70.2    | 81.0    | 63.6    | 62.6    | 71.9    | 64.1    | 52.4    | 76.1    | 68.9    | 55.4    | 73.6    |
| CMU     | 69.0 | 57.0    | 73.3    | 80.7    | 67.6    | 68.8    | 76.6    | 67.8    | 51.6    | 79.7    | 73.0    | 57.1    | 74.9    |

## Citation

If you use these methods in your research, please consider citing.

```
@inproceedings{CMU,
    title={Learning to detect open classes for universal domain adaptation},
    author={Fu, Bo and Cao, Zhangjie and Long, Mingsheng and Wang, Jianmin},
    booktitle={ECCV},
    year={2020}
}
```
