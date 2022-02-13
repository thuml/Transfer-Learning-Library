# Unsupervised Domain Adaptation for Image Regression Tasks
It’s suggested to use **pytorch==1.7.1** and torchvision==0.8.2 in order to better reproduce the benchmark results.

## Dataset

Following datasets can be downloaded automatically:

- [DSprites](https://github.com/deepmind/dsprites-dataset)
- [MPI3D](https://github.com/rr-learning/disentanglement_dataset)

## Supported Methods

Supported methods include:

- [Disparity Discrepancy (DD)](https://arxiv.org/abs/1904.05801)
- [Representation Subspace Distance (RSD)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/Representation-Subspace-Distance-for-Domain-Adaptation-Regression-icml21.pdf)

## Experiment and Results

The shell files give the script to reproduce the benchmark results with specified hyper-parameters.
For example, if you want to train DD on DSprites, use the following script

```shell script
# Train a DD on DSprites C->N task using ResNet 18.
# Assume you have put the datasets under the path `data/dSprites`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python dd.py data/dSprites -d DSprites -s C -t N -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_C2N --wd 0.0005
```

**Notations**

- ``Origin`` means the accuracy reported by the original paper.
- ``Avg`` is the accuracy reported by Transfer-Learn.
- ``ERM`` refers to the model trained with data from the source domain.
- ``Oracle`` refers to the model trained with data from the target domain.

Labels are all normalized to [0, 1] to eliminate the effects of diverse scale in regression values.

We repeat experiments on DD for three times and report the average error of the ``final`` epoch.


### dSprites error on ResNet-18

| Methods     | Avg   | C → N | C → S | N → C | N → S | S → C | S → N |
|-------------|-------|-------|-------|-------|-------|-------|-------|
| ERM | 0.157 | 0.232 | 0.271 | 0.081 | 0.22  | 0.038 | 0.092 |
| DD          | 0.057 | 0.047 | 0.08  | 0.03  | 0.095 | 0.053 | 0.037 |

### MPI3D error on ResNet-18

| Methods     | Avg   | RL → RC | RL → T | RC → RL | RC → T | T → RL | T → RC |
|-------------|-------|---------|--------|---------|--------|--------|--------|
| ERM | 0.176 | 0.232   | 0.271  | 0.081   | 0.22   | 0.038  | 0.092  |
| DD          | 0.03  | 0.086   | 0.029  | 0.057   | 0.189  | 0.131  | 0.087  |

## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{MDD,
    title={Bridging theory and algorithm for domain adaptation},
    author={Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael},
    booktitle={ICML},
    year={2019},
}

@inproceedings{RSD,
    title={Representation Subspace Distance for Domain Adaptation Regression},  
    author={Chen, Xinyang and Wang, Sinan and Wang, Jianmin and Long, Mingsheng}, 
    booktitle={ICML}, 
    year={2021} 
}
```
