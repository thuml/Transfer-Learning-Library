#!/usr/bin/env bash

# Ranking Pre-trained Model
# ======================================================================================================================
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a resnet50 --metric LogME --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a resnet50 --metric HScore --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a resnet50 --metric LEEP --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a resnet50 --metric NCE --save_features

CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a densenet121 --metric LogME --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a densenet121 --metric HScore --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a densenet121 --metric LEEP --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a densenet121 --metric NCE --save_features

CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a googlenet --metric LogME --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a googlenet --metric HScore --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a googlenet --metric LEEP --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a googlenet --metric NCE --save_features

CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a mobilenet_v2 --metric LogME --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a mobilenet_v2 --metric HScore --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a mobilenet_v2 --metric LEEP --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/cifar100 -d CIFAR100  -a mobilenet_v2 --metric NCE --save_features


# ======================================================================================================================
# FGVCAircraft
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a resnet50 --metric LogME --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a resnet50 --metric HScore --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a resnet50 --metric LEEP --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a resnet50 --metric NCE --save_features

CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a densenet121 --metric LogME --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a densenet121 --metric HScore --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a densenet121 --metric LEEP --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a densenet121 --metric NCE --save_features

CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a googlenet --metric LogME --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a googlenet --metric HScore --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a googlenet --metric LEEP --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a googlenet --metric NCE --save_features

CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a mobilenet_v2 --metric LogME --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a mobilenet_v2 --metric HScore --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a mobilenet_v2 --metric LEEP --save_features
CUDA_VISIBLE_DEVICES=0 python ranking.py /data/finetune/FGVCAircraft -d Aircraft  -a mobilenet_v2 --metric NCE --save_features
