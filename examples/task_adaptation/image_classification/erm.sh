#!/usr/bin/env bash
# Supervised Pretraining
# CUB-200-2011
 CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/erm/cub200_100
 CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log logs/erm/cub200_50
 CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log logs/erm/cub200_30
 CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log logs/erm/cub200_15

# Standford Cars
 CUDA_VISIBLE_DEVICES=0 python erm.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --finetune --log logs/erm/car_100
 CUDA_VISIBLE_DEVICES=0 python erm.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --finetune --log logs/erm/car_50
 CUDA_VISIBLE_DEVICES=0 python erm.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --finetune --log logs/erm/car_30
 CUDA_VISIBLE_DEVICES=0 python erm.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --finetune --log logs/erm/car_15

# Aircrafts
 CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft -sr 100 --seed 0 --finetune --log logs/erm/aircraft_100
 CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft -sr 50 --seed 0 --finetune --log logs/erm/aircraft_50
 CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft -sr 30 --seed 0 --finetune --log logs/erm/aircraft_30
 CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft -sr 15 --seed 0 --finetune --log logs/erm/aircraft_15

# CIFAR10
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar10 -d CIFAR10 --seed 0 --finetune --log logs/erm/cifar10/1e-2 --lr 1e-2
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python erm.py data/cifar100 -d CIFAR100 --seed 0 --finetune --log logs/erm/cifar100/1e-2 --lr 1e-2
# Flowers
CUDA_VISIBLE_DEVICES=0 python erm.py data/oxford_flowers102 -d OxfordFlowers102 --seed 0 --finetune --log logs/erm/oxford_flowers102/1e-2 --lr 1e-2
# Pets
CUDA_VISIBLE_DEVICES=0 python erm.py data/oxford_pet -d OxfordIIITPets --seed 0 --finetune --log logs/erm/oxford_pet/1e-2 --lr 1e-2
# DTD
CUDA_VISIBLE_DEVICES=0 python erm.py data/dtd -d DTD --seed 0 --finetune --log logs/erm/dtd/1e-2 --lr 1e-2
# caltech101
CUDA_VISIBLE_DEVICES=0 python erm.py data/caltech101 -d Caltech101 --seed 0 --finetune --log logs/erm/caltech101/lr_1e-3 --lr 1e-3
# SUN397
CUDA_VISIBLE_DEVICES=0 python erm.py data/sun397 -d SUN397 --seed 0 --finetune --log logs/erm/sun397/lr_1e-2 --lr 1e-2
# Food 101
CUDA_VISIBLE_DEVICES=0 python erm.py data/food-101 -d Food101 --seed 0 --finetune --log logs/erm/food-101/lr_1e-2 --lr 1e-2
# Standford Cars
CUDA_VISIBLE_DEVICES=0 python erm.py data/stanford_cars -d StanfordCars --seed 0 --finetune --log logs/erm/stanford_cars/lr_1e-2 --lr 1e-2
# Standford Cars
CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft --seed 0 --finetune --log logs/erm/aircraft/lr_1e-2 --lr 1e-2

# MoCo (Unsupervised Pretraining)
#CUB-200-2011
 CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/cub200_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/cub200_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python erm.py data/cub200 -d CUB200 -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/cub200_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Standford Cars
 CUDA_VISIBLE_DEVICES=0 python erm.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/cars_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python erm.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/cars_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python erm.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/cars_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python erm.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/cars_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Aircrafts
 CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/aircraft_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/aircraft_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/aircraft_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python erm.py data/aircraft -d Aircraft -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_erm/aircraft_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth
