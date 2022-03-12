#!/usr/bin/env bash

# Ranking Pre-trained Model
# ======================================================================================================================
# CIFAR10
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/cifar10 -d CIFAR10  -a resnet50 --save_features
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/cifar10 -d CIFAR10  -a densenet121  --save_features
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/cifar10 -d CIFAR10  -a googlenet  --save_features
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/cifar10 -d CIFAR10  -a mobilenet_v2  --save_features


# ======================================================================================================================
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/cifar100 -d CIFAR100  -a resnet50 --save_features
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/cifar100 -d CIFAR100  -a densenet121  --save_features
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/cifar100 -d CIFAR100  -a googlenet  --save_features
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/cifar100 -d CIFAR100  -a mobilenet_v2  --save_features

# ======================================================================================================================
# FGVCAircraft
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/FGVCAircraft -d Aircraft  -a resnet50  --save_features
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/FGVCAircraft -d Aircraft  -a densenet121  --save_features
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/FGVCAircraft -d Aircraft  -a googlenet  --save_features
CUDA_VISIBLE_DEVICES=0 python leep.py ./data/FGVCAircraft -d Aircraft  -a mobilenet_v2  --save_features
