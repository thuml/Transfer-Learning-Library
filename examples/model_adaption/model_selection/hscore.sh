#!/usr/bin/env bash

# Ranking Pre-trained Model
# ======================================================================================================================
# CIFAR10
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a mobilenet_v2 -l classifier[-1] --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar10 -d CIFAR10 -a mnasnet1_0 -l classifier[-1] --save_features

# ======================================================================================================================
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a mobilenet_v2 -l classifier[-1] --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/cifar100 -d CIFAR100 -a mnasnet1_0 -l classifier[-1] --save_features

# ======================================================================================================================
# FGVCAircraft
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a mobilenet_v2 -l classifier[-1] --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/FGVCAircraft -d Aircraft -a mnasnet1_0 -l classifier[-1] --save_features

# ======================================================================================================================
# Caltech101
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a mobilenet_v2 -l classifier[-1] --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/caltech101 -d Caltech101 -a mnasnet1_0 -l classifier[-1] --save_features

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a mobilenet_v2 -l classifier[-1] --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/dtd -d DTD -a mnasnet1_0 -l classifier[-1] --save_features

# ======================================================================================================================
# Oxford-IIIT
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a mobilenet_v2 -l classifier[-1] --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/Oxford-IIIT -d OxfordIIITPets -a mnasnet1_0 -l classifier[-1] --save_features

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a mobilenet_v2 -l classifier[-1] --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/stanford_cars -d StanfordCars -a mnasnet1_0 -l classifier[-1] --save_features

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a mobilenet_v2 -l classifier[-1] --save_features
CUDA_VISIBLE_DEVICES=0 python hscore.py ./data/SUN397 -d SUN397 -a mnasnet1_0 -l classifier[-1] --save_features
