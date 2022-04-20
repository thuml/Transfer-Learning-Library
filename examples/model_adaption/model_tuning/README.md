# Model Tuning

## Installation
Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

- [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
- [CIFAR10](http://www.cs.utoronto.ca/~kriz/cifar.html)
- [CIFAR100](http://www.cs.utoronto.ca/~kriz/cifar.html)
- [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html)
- [OxfordIIITPets](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [SUN397](https://vision.princeton.edu/projects/2010/SUN/)

## Supported Methods

Supported methods include:

- [Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs (B-Tuning, arxiv)](https://arxiv.org/pdf/2110.10545v1.pdf)

- [Zoo-Tuning: Adaptive Transfer from a Zoo of Models (Zoo-Tuning, ICML 2021)](https://arxiv.org/pdf/2106.15434.pdf)

- [Distilling the Knowledge in a Neural Network (Distill, arXiv 2015)](https://arxiv.org/pdf/2102.11005.pdf)


## Experiment and Results
 
 
## TODO


## Citation
If you use these methods in your research, please consider citing.

```
@article{you2021ranking,
  title={Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs},
  author={You, Kaichao and Liu, Yong and Wang, Jianmin and Jordan, Michael I and Long, Mingsheng},
  journal={arXiv preprint arXiv:2110.10545},
  year={2021}
}

@inproceedings{shu2021zoo,
  title={Zoo-tuning: Adaptive transfer from a zoo of models},
  author={Shu, Yang and Kou, Zhi and Cao, Zhangjie and Wang, Jianmin and Long, Mingsheng},
  booktitle={International Conference on Machine Learning},
  pages={9626--9637},
  year={2021},
  organization={PMLR}
}

@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff and others}
}

```