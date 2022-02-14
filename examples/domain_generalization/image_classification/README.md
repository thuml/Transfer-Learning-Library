# Domain Generalization for Image Classification

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
- [DomainNet](http://ai.bu.edu/M3SDA/)
- [PACS](https://domaingeneralization.github.io/#data)

## Supported Methods

- [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net (IBN-Net, 2018 ECCV)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf)
- [Domain Generalization with MixStyle (MixStyle, 2021 ICLR)](https://arxiv.org/abs/2104.02008)
- [Learning to Generalize: Meta-Learning for Domain Generalization (MLDG, 2018 AAAI)](https://arxiv.org/pdf/1710.03463.pdf)
- [Invariant Risk Minimization (IRM)](https://arxiv.org/abs/1907.02893)
- [Out-of-Distribution Generalization via Risk Extrapolation (VREx, 2021 ICML)](https://arxiv.org/abs/2003.00688)
- [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization (GroupDRO)](https://arxiv.org/abs/1911.08731)
- [Deep CORAL: Correlation Alignment for Deep Domain Adaptation (Deep Coral, 2016 ECCV)](https://arxiv.org/abs/1607.01719)

## Usage

The shell files give the script to reproduce the benchmark with specified hyper-parameters.
For example, if you want to train IRM on Office-Home, use the following script

```shell script
# Train with IRM on Office-Home Ar Cl Rw -> Pr task using ResNet 50.
# Assume you have put the datasets under the path `data/office-home`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python irm.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/irm/OfficeHome_Pr
```
Note that ``-s`` specifies the source domain, ``-t`` specifies the target domain,
and ``--log`` specifies where to store results.

## Experiment and Results
Following [DomainBed](https://github.com/facebookresearch/DomainBed), we select hyper-parameters based on
the model's performance on `training-domain validation set` (first rule in DomainBed).
Concretely, we save model with the highest accuracy on `training-domain validation set` and then 
load this checkpoint to test on the target domain.

Here are some differences between our implementation and DomainBed. For the model, 
we do not freeze `BatchNorm2d` layers and do not insert additional `Dropout` layer except for `PACS` dataset. 
For the optimizer, we use `SGD` with momentum by default and find this usually achieves better performance than `Adam`.

**Notations**
- ``ERM`` refers to the model trained with data from the source domain.
- ``Avg`` is the accuracy reported by `TLlib`.

### PACS accuracy on ResNet-50

| Methods  | avg  | A    | C    | P    | S    |
|----------|------|------|------|------|------|
| ERM      | 86.4 | 88.5 | 78.4 | 97.2 | 81.4 |
| IBN      | 87.8 | 88.2 | 84.5 | 97.1 | 81.4 |
| MixStyle | 87.4 | 87.8 | 82.3 | 95.0 | 84.5 |
| MLDG     | 87.2 | 88.2 | 81.4 | 96.6 | 82.5 |
| IRM      | 86.9 | 88.0 | 82.5 | 98.0 | 79.0 |
| VREx     | 87.0 | 87.2 | 82.3 | 97.4 | 81.0 |
| GroupDRO | 87.3 | 88.9 | 81.7 | 97.8 | 80.8 |
| CORAL    | 86.4 | 89.1 | 80.0 | 97.4 | 79.1 |

### Office-Home accuracy on ResNet-50

| Methods  | avg  | A    | C    | P    | R    |
|----------|------|------|------|------|------|
| ERM      | 70.8 | 68.3 | 55.9 | 78.9 | 80.0 |
| IBN      | 69.9 | 67.4 | 55.2 | 77.3 | 79.6 |
| MixStyle | 71.7 | 66.8 | 58.1 | 78.0 | 79.9 |
| MLDG     | 70.3 | 65.9 | 57.6 | 78.2 | 79.6 |
| IRM      | 70.3 | 66.7 | 54.8 | 78.6 | 80.9 |
| VREx     | 70.2 | 66.9 | 54.9 | 78.2 | 80.9 |
| GroupDRO | 70.0 | 66.7 | 55.2 | 78.8 | 79.9 |
| CORAL    | 70.9 | 68.3 | 55.4 | 78.8 | 81.0 |

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

@inproceedings{MLDG,
    title={Learning to Generalize: Meta-Learning for Domain Generalization},
    author={Li, Da and Yang, Yongxin and Song, Yi-Zhe and Hospedales, Timothy},
    booktitle={AAAI},
    year={2018}
}
 
@misc{IRM,
    title={Invariant Risk Minimization}, 
    author={Martin Arjovsky and Léon Bottou and Ishaan Gulrajani and David Lopez-Paz},
    year={2020},
    eprint={1907.02893},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}

@inproceedings{VREx,
    title={Out-of-Distribution Generalization via Risk Extrapolation (REx)}, 
    author={David Krueger and Ethan Caballero and Joern-Henrik Jacobsen and Amy Zhang and Jonathan Binas and Dinghuai Zhang and Remi Le Priol and Aaron Courville},
    year={2021},
    booktitle={ICML},
}

@inproceedings{GroupDRO,
    title={Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization}, 
    author={Shiori Sagawa and Pang Wei Koh and Tatsunori B. Hashimoto and Percy Liang},
    year={2020},
    booktitle={ICLR}
}

@inproceedings{deep_coral,
    title={Deep coral: Correlation alignment for deep domain adaptation},
    author={Sun, Baochen and Saenko, Kate},
    booktitle={ECCV},
    year={2016},
}
```