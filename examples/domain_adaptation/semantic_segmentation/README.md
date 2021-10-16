# Unsupervised Domain Adaptation for Semantic Segmentation

## Dataset

You need to prepare following datasets manually if you want to use them:
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/)
- [Synthia](https://synthia-dataset.net/)

and prepare them following [Documentations for Cityscapes](/common/vision/datasets/segmentation/cityscapes.py), [GTA5](/common/vision/datasets/segmentation/gta5.py) and [Synthia](/common/vision/datasets/segmentation/synthia.py), 

## Supported Methods

Supported methods include:

- [Cycle-Consistent Adversarial Networks (CycleGAN)](https://arxiv.org/pdf/1703.10593.pdf)
- [CyCADA: Cycle-Consistent Adversarial Domain Adaptation](https://arxiv.org/abs/1711.03213)
- [Adversarial Entropy Minimization (ADVENT)](https://arxiv.org/abs/1811.12833)
- [Fourier Domain Adaptation (FDA)](https://arxiv.org/abs/2004.05498)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/semantic_segmentation.rst) with specified hyper-parameters.
For example, if you want to train ADVENT on Office31, use the following script

```shell script
# Train a ADVENT on GTA5 to Cityscapes.
# Assume you have put the datasets under the path `data/GTA5` and `data/Cityscapes`, 
CUDA_VISIBLE_DEVICES=0 python advent.py data/GTA5 data/Cityscapes -s GTA5 -t Cityscapes \
    --log logs/advent/gtav2cityscapes
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.

## TODO
Support methods: AdaptSeg

## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{CycleGAN,
    title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
    author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
    booktitle={ICCV},
    year={2017}
}

@inproceedings{cycada,
    title={Cycada: Cycle-consistent adversarial domain adaptation},
    author={Hoffman, Judy and Tzeng, Eric and Park, Taesung and Zhu, Jun-Yan and Isola, Phillip and Saenko, Kate and Efros, Alexei and Darrell, Trevor},
    booktitle={ICML},
    year={2018},
}

@inproceedings{Advent,
    author = {Vu, Tuan-Hung and Jain, Himalaya and Bucher, Maxime and Cord, Matthieu and Perez, Patrick},
    title = {ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation},
    booktitle = {CVPR},
    year = {2019}
}

@inproceedings{FDA,
    author    = {Yanchao Yang and
               Stefano Soatto},
    title     = {{FDA:} Fourier Domain Adaptation for Semantic Segmentation},
    booktitle = {CVPR},
    year = {2020}
}
```
