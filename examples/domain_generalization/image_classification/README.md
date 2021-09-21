# Domain Generalization for Image Classification

## Installation
Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

Following datasets can be downloaded automatically:

- OfficeHome
- DomainNet

## Supported Methods

- Instance-Batch Normalization Network (IBN-Net)
- Domain Generalization with MixStyle (MixStyle)
- Meta Learning for Domain Generalization (MLDG)
- Invariant Risk Minimization (IRM)
- Variance Risk Extrapolation (VREx)
- Group Distributionally robust optimization (GroupDRO)
- Correlation Alignment for Deep Domain Adaptation (Deep CORAL)

The shell files give the script to reproduce the [benchmarks](/docs/dglib/benchmarks/classification.rst) with specified hyper-parameters.
For example, if you want to reproduce IRM on Office-Home, use the following script

```shell script
# Train with IRM on Office-Home Ar Cl Rw -> Pr task using ResNet 50.
# Assume you have put the datasets under the path `data/office-home`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python irm.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/irm/OfficeHome_Pr
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.
