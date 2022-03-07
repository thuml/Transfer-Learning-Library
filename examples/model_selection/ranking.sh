#!/usr/bin/env bash

# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# Food 101
CUDA_VISIBLE_DEVICES=0 python ranking.py data/food101 -d Food101 --num-samples-per-class 4 --finetune --lr 0.01 \
  -a resnet50 --seed 0 --log logs/erm/food101_4_labels_per_class

