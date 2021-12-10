# Unsupervised Domain Adaptation for Object Detection

## Installation
Our code is based on [Detectron latest(v0.6)](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), please install it before usage.

## Dataset

You need to prepare following datasets manually if you want to use them:
- [Cityscapes](https://www.cityscapes-dataset.com/)


## Supported Methods

Supported methods include:

- [Cycle-Consistent Adversarial Networks (CycleGAN)](https://arxiv.org/pdf/1703.10593.pdf)
- [Decoupled Adaptation for Cross-Domain Object Detection (D-adapt)](https://arxiv.org/abs/2110.02578)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/object_detection.rst) with specified hyper-parameters.

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.

## TODO
Support methods: SWDA

## Citation
If you use these methods in your research, please consider citing.

```
@misc{jiang2021decoupled,
      title={Decoupled Adaptation for Cross-Domain Object Detection}, 
      author={Junguang Jiang and Baixu Chen and Jianmin Wang and Mingsheng Long},
      year={2021},
      eprint={2110.02578},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{CycleGAN,
    title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
    author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
    booktitle={ICCV},
    year={2017}
}
```
