# Domain Generalization for Image Classification

## Installation
Example scripts can deal with [WILDS datasets](https://wilds.stanford.edu/).
You should first install ``wilds`` before using these scripts.

```
pip install wilds
```

Example scripts also support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
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
- [iwildcam (WILDS)](https://wilds.stanford.edu/datasets/)
- [camelyon17 (WILDS)](https://wilds.stanford.edu/datasets/)
- [fmow (WILDS)](https://wilds.stanford.edu/datasets/)

## Supported Methods

- [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net (IBN-Net, 2018 ECCV)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf)
- [Domain Generalization with MixStyle (MixStyle, 2021 ICLR)](https://arxiv.org/abs/2104.02008)
- [Learning to Generalize: Meta-Learning for Domain Generalization (MLDG, 2018 AAAI)](https://arxiv.org/pdf/1710.03463.pdf)
- [Invariant Risk Minimization (IRM)](https://arxiv.org/abs/1907.02893)
- [Out-of-Distribution Generalization via Risk Extrapolation (VREx, 2021 ICML)](https://arxiv.org/abs/2003.00688)
- [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization (GroupDRO)](https://arxiv.org/abs/1911.08731)
- [Deep CORAL: Correlation Alignment for Deep Domain Adaptation (Deep Coral, 2016 ECCV)](https://arxiv.org/abs/1607.01719)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dglib/benchmarks/image_classification.rst) with specified hyper-parameters.
For example, if you want to reproduce IRM on Office-Home, use the following script

```shell script
# Train with IRM on Office-Home Ar Cl Rw -> Pr task using ResNet 50.
# Assume you have put the datasets under the path `data/office-home`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python irm.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/irm/OfficeHome_Pr
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