#!/usr/bin/env bash

# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/food101 -d Food101 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/pseudo_label/food101_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/food101 -d Food101 --num-samples-per-class 10 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/pseudo_label/food101_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/food101 -d Food101 --oracle --finetune --lr 0.01 \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/food101_oracle

# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/cifar10_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/cifar10_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/cifar10_oracle

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/pseudo_label/cifar100_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 10 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/pseudo_label/cifar100_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --oracle --finetune --lr 0.01 \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/cifar100_oracle

# ======================================================================================================================
# CUB 200
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/pseudo_label/cub200_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/pseudo_label/cub200_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 --oracle --finetune \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/cub200_oracle

# ======================================================================================================================
# Aircraft
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/aircraft_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/aircraft_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/aircraft_oracle

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cars -d StanfordCars --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/car_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cars -d StanfordCars --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/car_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cars -d StanfordCars --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/car_oracle

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/sun397 -d SUN397 --num-samples-per-class 4 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/pseudo_label/sun_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/sun397 -d SUN397 --num-samples-per-class 10 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/pseudo_label/sun_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/sun397 -d SUN397 --oracle --finetune --lr 0.001 \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/sun_oracle

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/dtd -d DTD --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/dtd_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/dtd -d DTD --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/dtd_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/dtd -d DTD --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/dtd_oracle

# ======================================================================================================================
# Oxford Pets
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/pets -d OxfordIIITPets --num-samples-per-class 4 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/pseudo_label/pets_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/pets -d OxfordIIITPets --num-samples-per-class 10 --finetune --lr 0.001 \
  -a resnet50 --seed 0 --log logs/pseudo_label/pets_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/pets -d OxfordIIITPets --oracle --finetune --lr 0.001 \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/pets_oracle

# ======================================================================================================================
# Oxford Flowers
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/flowers -d OxfordFlowers102 --num-samples-per-class 4 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/flowers_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/flowers -d OxfordFlowers102 --num-samples-per-class 10 --finetune --lr 0.03 \
  -a resnet50 --seed 0 --log logs/pseudo_label/flowers_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/flowers -d OxfordFlowers102 --oracle --finetune --lr 0.03 \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/flowers_oracle

# ======================================================================================================================
# Caltech 101
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/caltech101 -d Caltech101 --num-samples-per-class 4 --finetune \
  -a resnet50 --seed 0 --log logs/pseudo_label/caltech_4_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/caltech101 -d Caltech101 --num-samples-per-class 10 --finetune \
  -a resnet50 --seed 0 --log logs/pseudo_label/caltech_10_labels_per_class
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/caltech101 -d Caltech101 --oracle --finetune \
  -a resnet50 --epochs 80 --seed 0 --log logs/pseudo_label/caltech_oracle
