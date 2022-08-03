# Open-set Domain Adaptation for Image Classification

## Installation
It’s suggested to use **pytorch==1.7.1** and torchvision==0.8.2 in order to reproduce the benchmark results.

Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [Office31](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [VisDA2017](http://ai.bu.edu/visda-2017/)

## Supported Methods

Supported methods include:

- [Open Set Domain Adaptation (OSBP)](https://arxiv.org/abs/1804.10427)

## Experiment and Results

The shell files give the script to reproduce the benchmark with specified hyper-parameters.
For example, if you want to train DANN on Office31, use the following script

```shell script
# Train a DANN on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
```

**Notations**
- ``Origin`` means the accuracy reported by the original paper.
- ``Avg`` is the accuracy reported by `TLlib`.
- ``ERM`` refers to the model trained with data from the source domain.

We report ``HOS`` used in [ROS (ECCV 2020)](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610409.pdf) to better measure the abilities of different open set domain adaptation algorithms.

We report the best ``HOS`` in all epochs.
DANN (baseline model) will degrade performance as training progresses, thus the
final ``HOS`` will be much lower than reported.
In contrast, OSBP will improve performance stably.


### Office-31 H-Score on ResNet-50

| Methods     | Avg  | A → W | D → W | W → D | A → D | D → A | W → A |
|-------------|------|-------|-------|-------|-------|-------|-------|
| ERM         | 75.9 | 67.7  | 85.7  | 91.4  | 72.1  | 68.4  | 67.8  |
| DANN        | 80.4 | 81.4  | 89.1  | 92.0  | 82.5  | 66.7  | 70.4  |
| OSBP        | 87.8 | 90.7  | 96.4  | 97.5  | 88.7  | 77.0  | 76.7  |

### Office-Home HOS on ResNet-50

| Methods     | Origin | Avg  | Ar → Cl | Ar → Pr | Ar → Rw | Cl → Ar | Cl → Pr | Cl → Rw | Pr → Ar | Pr → Cl | Pr → Rw | Rw → Ar | Rw → Cl | Rw → Pr |
|-------------|--------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Source Only | /      | 59.8 | 55.2    | 65.2    | 71.4    | 52.8    | 59.6    | 65.2    | 55.8    | 44.8    | 68.0    | 63.8    | 49.4    | 68.0    |
| DANN        | /      | 64.8 | 55.2    | 65.2    | 71.4    | 52.8    | 59.6    | 65.2    | 55.8    | 44.8    | 68.0    | 63.8    | 49.4    | 68.0    |
| OSBP        | 64.7   | 68.6 | 62.0    | 70.8    | 76.5    | 66.4    | 68.8    | 73.8    | 65.8    | 57.1    | 75.4    | 70.6    | 60.6    | 75.9    |

### VisDA-2017 performance on ResNet-50
| Methods     | HOS  | OS   | OS*  | UNK  | bcycl | bus  | car  | mcycl | train | truck |
|-------------|------|------|------|------|-------|------|------|-------|-------|-------|
| Source Only | 42.6 | 37.6 | 34.7 | 55.1 | 42.6  | 6.4  | 30.5 | 67.1  | 84.0  | 0.2   |
| DANN        | 57.8 | 50.4 | 45.6 | 78.9 | 20.1  | 71.4 | 29.5 | 74.4  | 67.8  | 10.4  |
| OSBP        | 75.4 | 67.3 | 62.9 | 94.3 | 63.7  | 75.9 | 49.6 | 74.4  | 86.2  | 27.3  |

## Citation
If you use these methods in your research, please consider citing.

```
@InProceedings{OSBP,
    author = {Saito, Kuniaki and Yamamoto, Shohei and Ushiku, Yoshitaka and Harada, Tatsuya},
    title = {Open Set Domain Adaptation by Backpropagation},
    booktitle = {ECCV},
    year = {2018}
}
```
