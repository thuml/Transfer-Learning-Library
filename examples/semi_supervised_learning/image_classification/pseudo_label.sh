#!/usr/bin/env bash

# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/food101 -d Food101 --num-samples-per-class 4 -a resnet50 \
  --lr 0.03 --finetune --threshold 0.95 --seed 0 --log logs/pseudo_label/food101_4_labels_per_class

# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.8 --seed 0 --log logs/pseudo_label/cifar10_4_labels_per_class

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.8 --seed 0 --log logs/pseudo_label/cifar100_4_labels_per_class

# ======================================================================================================================
# CUB 200
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --threshold 0.8 --seed 0 --log logs/pseudo_label/cub200_4_labels_per_class

# ======================================================================================================================
# Aircraft
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft --num-samples-per-class 4 -a resnet50 \
  --lr 0.03 --finetune --threshold 0.95 --seed 0 --log logs/pseudo_label/aircraft_4_labels_per_class

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cars -d StanfordCars --num-samples-per-class 4 -a resnet50 \
  --lr 0.03 --finetune --threshold 0.95 --seed 0 --log logs/pseudo_label/car_4_labels_per_class

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/sun397 -d SUN397 --num-samples-per-class 4 -a resnet50 \
  --lr 0.001 --finetune --threshold 0.7 --seed 0 --log logs/pseudo_label/sun_4_labels_per_class

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/dtd -d DTD --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --threshold 0.95 --seed 0 --log logs/pseudo_label/dtd_4_labels_per_class

# ======================================================================================================================
# Oxford Pets
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/pets -d OxfordIIITPets --num-samples-per-class 4 -a resnet50 \
  --lr 0.03 --finetune --threshold 0.95 --seed 0 --log logs/pseudo_label/pets_4_labels_per_class

# ======================================================================================================================
# Oxford Flowers
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/flowers -d OxfordFlowers102 --num-samples-per-class 4 -a resnet50 \
  --lr 0.03 --finetune --threshold 0.95 --seed 0 --log logs/pseudo_label/flowers_4_labels_per_class

# ======================================================================================================================
# Caltech 101
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/caltech101 -d Caltech101 --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.7 --seed 0 --log logs/pseudo_label/caltech_4_labels_per_class

# ImageNet Unsupervised Pretrain (MoCov2, ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/food101 -d Food101 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.003 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/food101_4_labels_per_class

# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/cifar10_4_labels_per_class

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.003 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/cifar100_4_labels_per_class

# ======================================================================================================================
# CUB 200
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.01 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/cub200_4_labels_per_class

# ======================================================================================================================
# Aircraft
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.01 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/aircraft_4_labels_per_class

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cars -d StanfordCars --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.03 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/car_4_labels_per_class

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/sun397 -d SUN397 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.003 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/sun_4_labels_per_class

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/dtd -d DTD --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/dtd_4_labels_per_class

# ======================================================================================================================
# Oxford Pets
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/pets -d OxfordIIITPets --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --threshold 0.7 --seed 0 --log logs/pseudo_label_moco_pretrain/pets_4_labels_per_class

# ======================================================================================================================
# Oxford Flowers
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/flowers -d OxfordFlowers102 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.03 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/flowers_4_labels_per_class

# ======================================================================================================================
# Caltech 101
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/caltech101 -d Caltech101 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.003 --finetune --lr-scheduler cos --threshold 0.95 --seed 0 --log logs/pseudo_label_moco_pretrain/caltech_4_labels_per_class
