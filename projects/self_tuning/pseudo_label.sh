#!/usr/bin/env bash
# Supervised Pretraining
# ResNet50, CUB200
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/pseudo_label/cub200_15
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 -sr 30 --seed 0 --log logs/pseudo_label/cub200_30
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 -sr 50 --seed 0 --log logs/pseudo_label/cub200_50

# ResNet50, StanfordCars
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/pseudo_label/car_15
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/pseudo_label/car_30
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/pseudo_label/car_50

# ResNet50, Aircraft
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/pseudo_label/aircraft_15
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/pseudo_label/aircraft_30
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/pseudo_label/aircraft_50

# EfficientNet-B2, CIFAR100
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 \
  -a efficientnet_b2 -b 40 --num-samples 400 --seed 0 --log logs/pseudo_label/cifar100_400

CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 \
  -a efficientnet_b2 -b 40 --num-samples 2500 --seed 0 --log logs/pseudo_label/cifar100_2500

CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 \
  -a efficientnet_b2 -b 40 --num-samples 10000 --seed 0 --log logs/pseudo_label/cifar100_10000

# MoCo (Unsupervised Pretraining)
# ResNet50, CUB200
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 15 --seed 0 --log logs/pseudo_label_moco/cub200_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 30 --seed 0 --log logs/pseudo_label_moco/cub200_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/cub200 -d CUB200 --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 50 --seed 0 --log logs/pseudo_label_moco/cub200_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# ResNet50, StanfordCars
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/stanford_cars -d StanfordCars --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 15 --seed 0 --log logs/pseudo_label_moco/car_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/stanford_cars -d StanfordCars --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 30 --seed 0 --log logs/pseudo_label_moco/car_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/stanford_cars -d StanfordCars --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 50 --seed 0 --log logs/pseudo_label_moco/car_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# ResNet50, Aircraft
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 15 --seed 0 --log logs/pseudo_label_moco/aircraft_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 30 --seed 0 --log logs/pseudo_label_moco/aircraft_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py data/aircraft -d Aircraft --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 50 --seed 0 --log logs/pseudo_label_moco/aircraft_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
