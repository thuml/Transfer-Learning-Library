#!/usr/bin/env bash

# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python dst.py data/food101 -d Food101 --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --threshold 0.8 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/dst/food101_4_labels_per_class

# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python dst.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.7 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/dst/cifar10_4_labels_per_class

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python dst.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.7 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/dst/cifar100_4_labels_per_class

# ======================================================================================================================
# CUB 200
CUDA_VISIBLE_DEVICES=0 python dst.py data/cub200 -d CUB200 --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --threshold 0.95 --trade-off-self-training 0.3 --eta-prime 2 \
  --seed 0 --log logs/dst/cub200_4_labels_per_class

# ======================================================================================================================
# Aircraft
CUDA_VISIBLE_DEVICES=0 python dst.py data/aircraft -d Aircraft --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --threshold 0.7 --trade-off-self-training 1 --eta-prime 1 \
  --seed 0 --log logs/dst/aircraft_4_labels_per_class

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python dst.py data/cars -d StanfordCars --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --threshold 0.7 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/dst/car_4_labels_per_class

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python dst.py data/sun397 -d SUN397 --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --threshold 0.7 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/dst/sun_4_labels_per_class

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python dst.py data/dtd -d DTD --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.95 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/dst/dtd_4_labels_per_class

# ======================================================================================================================
# Oxford Pets
CUDA_VISIBLE_DEVICES=0 python dst.py data/pets -d OxfordIIITPets --num-samples-per-class 4 -a resnet50 \
  --lr 0.001 --finetune --threshold 0.9 --trade-off-self-training 0.3 --eta-prime 2 \
  --seed 0 --log logs/dst/pets_4_labels_per_class

# ======================================================================================================================
# Oxford Flowers
CUDA_VISIBLE_DEVICES=0 python dst.py data/flowers -d OxfordFlowers102 --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --threshold 0.9 --trade-off-self-training 0.3 --eta-prime 1 \
  --seed 0 --log logs/dst/flowers_4_labels_per_class

# ======================================================================================================================
# Caltech 101
CUDA_VISIBLE_DEVICES=0 python dst.py data/caltech101 -d Caltech101 --num-samples-per-class 4 -a resnet50 \
  --lr 0.001 --finetune --threshold 0.7 --trade-off-self-training 1 --eta-prime 4 \
  --seed 0 --log logs/dst/caltech_4_labels_per_class

# ImageNet Unsupervised Pretrain (MoCov2, ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python dst.py data/food101 -d Food101 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.003 --finetune --lr-scheduler cos --threshold 0.7 --trade-off-self-training 1 --eta-prime 1 \
  --seed 0 --log logs/dst_moco_pretrain/food101_4_labels_per_class

# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python dst.py data/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --threshold 0.95 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/dst_moco_pretrain/cifar10_4_labels_per_class

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python dst.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --threshold 0.7 --trade-off-self-training 1 --eta-prime 1 \
  --seed 0 --log logs/dst_moco_pretrain/cifar100_4_labels_per_class

# ======================================================================================================================
# CUB 200
CUDA_VISIBLE_DEVICES=0 python dst.py data/cub200 -d CUB200 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.01 --finetune --lr-scheduler cos --threshold 0.7 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/dst_moco_pretrain/cub200_4_labels_per_class

# ======================================================================================================================
# Aircraft
CUDA_VISIBLE_DEVICES=0 python dst.py data/aircraft -d Aircraft --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.01 --finetune --lr-scheduler cos --threshold 0.7 --trade-off-self-training 1 --eta-prime 1 \
  --seed 0 --log logs/dst_moco_pretrain/aircraft_4_labels_per_class

# ======================================================================================================================
# StanfordCars
CUDA_VISIBLE_DEVICES=0 python dst.py data/cars -d StanfordCars --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.01 --finetune --lr-scheduler cos --threshold 0.7 --trade-off-self-training 1 --eta-prime 1 \
  --seed 0 --log logs/dst_moco_pretrain/car_4_labels_per_class

# ======================================================================================================================
# SUN397
CUDA_VISIBLE_DEVICES=0 python dst.py data/sun397 -d SUN397 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.003 --finetune --lr-scheduler cos --threshold 0.7 --trade-off-self-training 0.3 --eta-prime 2 \
  --seed 0 --log logs/dst_moco_pretrain/sun_4_labels_per_class

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python dst.py data/dtd -d DTD --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --threshold 0.95 --trade-off-self-training 0.1 --eta-prime 3 \
  --seed 0 --log logs/dst_moco_pretrain/dtd_4_labels_per_class

# ======================================================================================================================
# Oxford Pets
CUDA_VISIBLE_DEVICES=0 python dst.py data/pets -d OxfordIIITPets --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --threshold 0.7 --trade-off-self-training 0.1 --eta-prime 1 \
  --seed 0 --log logs/dst_moco_pretrain/pets_4_labels_per_class

# ======================================================================================================================
# Oxford Flowers
CUDA_VISIBLE_DEVICES=0 python dst.py data/flowers -d OxfordFlowers102 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.003 --finetune --lr-scheduler cos --threshold 0.95 --trade-off-self-training 1 --eta-prime 1 \
  --seed 0 --log logs/dst_moco_pretrain/flowers_4_labels_per_class

# ======================================================================================================================
# Caltech 101
CUDA_VISIBLE_DEVICES=0 python dst.py data/caltech101 -d Caltech101 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --threshold 0.95 --trade-off-self-training 0.1 --eta-prime 1 \
  --seed 0 --log logs/dst_moco_pretrain/caltech_4_labels_per_class
