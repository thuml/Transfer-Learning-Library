# Unsupervised Domain Adaptation for Image Regression Tasks

## Dataset

Following datasets can be downloaded automatically:

- [DSprites](https://github.com/deepmind/dsprites-dataset)
- [MPI3D](https://github.com/rr-learning/disentanglement_dataset)

## Supported Methods

Supported methods include:

- [Disparity Discrepancy (DD)](https://arxiv.org/abs/1904.05801)
- [Representation Subspace Distance (RSD)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/Representation-Subspace-Distance-for-Domain-Adaptation-Regression-icml21.pdf)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/image_regression.rst) with specified hyper-parameters.
For example, if you want to train DD on Office31, use the following script

```shell script
# Train a DD on DSprites C->N task using ResNet 18.
# Assume you have put the datasets under the path `data/dSprites`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python dd.py data/dSprites -d DSprites -s C -t N -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_C2N --wd 0.0005
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.

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
