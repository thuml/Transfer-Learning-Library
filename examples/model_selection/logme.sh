#!/usr/bin/env bash


# Ranking Pre-trained Model
# ======================================================================================================================
# CIFAR10
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar10 -d CIFAR10 -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar10 -d CIFAR10 -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar10 -d CIFAR10 -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar10 -d CIFAR10 -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar10 -d CIFAR10 -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar10 -d CIFAR10 -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar10 -d CIFAR10 -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar10 -d CIFAR10 -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar10 -d CIFAR10 -a mobilenet_v2 -l classifier[-1] --save_features

# ======================================================================================================================
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar100 -d CIFAR100 -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar100 -d CIFAR100 -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar100 -d CIFAR100 -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar100 -d CIFAR100 -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar100 -d CIFAR100 -a inception_v3 --resizing res.299 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar100 -d CIFAR100 -a densenet121 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar100 -d CIFAR100 -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar100 -d CIFAR100 -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/cifar100 -d CIFAR100 -a mobilenet_v2 -l classifier[-1] --save_features

# ======================================================================================================================
# FGVCAircraft
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a resnet50 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a resnet101 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a resnet152 -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a googlenet -l fc --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a inception_v3 --resizing res.299 -l fc --save_features &
CUDA_VISIBLE_DEVICES=1 python logme.py ./data/FGVCAircraft -d Aircraft -a densenet121 -l classifier --save_features &
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a densenet169 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a densenet201 -l classifier --save_features
CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a mobilenet_v2 -l classifier[-1] --save_features

