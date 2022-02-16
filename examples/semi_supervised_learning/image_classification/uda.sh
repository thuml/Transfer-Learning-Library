#!/usr/bin/env bash

# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python uda.py data/food101 -d Food101 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/uda/food101_4_labels_per_class

# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/cifar10_4_labels_per_class

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/uda/cifar100_4_labels_per_class

# ======================================================================================================================
# CUB 200
CUDA_VISIBLE_DEVICES=0 python uda.py data/cub200 -d CUB200 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/uda/cub200_4_labels_per_class

# ======================================================================================================================
# Aircraft
CUDA_VISIBLE_DEVICES=0 python uda.py data/aircraft -d Aircraft --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/aircraft_4_labels_per_class

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python uda.py data/cars -d StanfordCars --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/car_4_labels_per_class

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python uda.py data/sun397 -d SUN397 --num-samples-per-class 4 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/uda/sun_4_labels_per_class

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python uda.py data/dtd -d DTD --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/dtd_4_labels_per_class

# ======================================================================================================================
# Oxford Pets
CUDA_VISIBLE_DEVICES=0 python uda.py data/pets -d OxfordIIITPets --num-samples-per-class 4 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/uda/pets_4_labels_per_class

# ======================================================================================================================
# Oxford Flowers
CUDA_VISIBLE_DEVICES=0 python uda.py data/flowers -d OxfordFlowers102 --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/flowers_4_labels_per_class

# ======================================================================================================================
# Caltech 101
CUDA_VISIBLE_DEVICES=0 python uda.py data/caltech101 -d Caltech101 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/uda/caltech_4_labels_per_class
