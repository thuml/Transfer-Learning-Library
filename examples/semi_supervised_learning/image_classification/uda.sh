#!/usr/bin/env bash

# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python uda.py data/food101 -d Food101 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/uda/food101_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/food101 -d Food101 --num-samples-per-class 10 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/uda/food101_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/food101 -d Food101 --oracle --finetune --lr 0.01 \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/food101_oracle

# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/cifar10_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/cifar10_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/cifar10_oracle

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/uda/cifar100_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 10 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/uda/cifar100_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --oracle --finetune --lr 0.01 \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/cifar100_oracle

# ======================================================================================================================
# CUB 200
CUDA_VISIBLE_DEVICES=0 python uda.py data/cub200 -d CUB200 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/uda/cub200_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/cub200 -d CUB200 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/uda/cub200_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/cub200 -d CUB200 --oracle --finetune \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/cub200_oracle

# ======================================================================================================================
# Aircraft
CUDA_VISIBLE_DEVICES=0 python uda.py data/aircraft -d Aircraft --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/aircraft_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/aircraft -d Aircraft --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/aircraft_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/aircraft -d Aircraft --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/aircraft_oracle

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python uda.py data/cars -d StanfordCars --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/car_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/cars -d StanfordCars --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/car_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/cars -d StanfordCars --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/car_oracle

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python uda.py data/sun397 -d SUN397 --num-samples-per-class 4 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/uda/sun_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/sun397 -d SUN397 --num-samples-per-class 10 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/uda/sun_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/sun397 -d SUN397 --oracle --finetune --lr 0.001 \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/sun_oracle

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python uda.py data/dtd -d DTD --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/dtd_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/dtd -d DTD --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/dtd_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/dtd -d DTD --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/dtd_oracle

# ======================================================================================================================
# Oxford Pets
CUDA_VISIBLE_DEVICES=0 python uda.py data/pets -d OxfordIIITPets --num-samples-per-class 4 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/uda/pets_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/pets -d OxfordIIITPets --num-samples-per-class 10 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/uda/pets_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/pets -d OxfordIIITPets --oracle --finetune --lr 0.001 \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/pets_oracle

# ======================================================================================================================
# Oxford Flowers
CUDA_VISIBLE_DEVICES=0 python uda.py data/flowers -d OxfordFlowers102 --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/flowers_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/flowers -d OxfordFlowers102 --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/uda/flowers_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/flowers -d OxfordFlowers102 --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/flowers_oracle

# ======================================================================================================================
# Caltech 101
CUDA_VISIBLE_DEVICES=0 python uda.py data/caltech101 -d Caltech101 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/uda/caltech_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/caltech101 -d Caltech101 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/uda/caltech_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python uda.py data/caltech101 -d Caltech101 --oracle --finetune \
  -a resnet50 --epochs 80 --seed 0 --log logs/uda/caltech_oracle
