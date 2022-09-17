#!/usr/bin/env bash
# Supervised Pretraining
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/co_tuning/cub200_100
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log logs/co_tuning/cub200_50
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log logs/co_tuning/cub200_30
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log logs/co_tuning/cub200_15

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --finetune --log logs/co_tuning/car_100
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --finetune --log logs/co_tuning/car_50
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --finetune --log logs/co_tuning/car_30
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --finetune --log logs/co_tuning/car_15

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/aircraft -d Aircraft -sr 100 --seed 0 --finetune --log logs/co_tuning/aircraft_100
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/aircraft -d Aircraft -sr 50 --seed 0 --finetune --log logs/co_tuning/aircraft_50
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/aircraft -d Aircraft -sr 30 --seed 0 --finetune --log logs/co_tuning/aircraft_30
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/aircraft -d Aircraft -sr 15 --seed 0 --finetune --log logs/co_tuning/aircraft_15

# CIFAR10
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cifar10 -d CIFAR10 --seed 0 --finetune --log logs/co_tuning/cifar10/1e-2 --lr 1e-2
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cifar100 -d CIFAR100 --seed 0 --finetune --log logs/co_tuning/cifar100/1e-2 --lr 1e-2
# Flowers
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/oxford_flowers102 -d OxfordFlowers102 --seed 0 --finetune --log logs/co_tuning/oxford_flowers102/1e-2 --lr 1e-2
# Pets
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/oxford_pet -d OxfordIIITPets --seed 0 --finetune --log logs/co_tuning/oxford_pet/1e-2 --lr 1e-2
# DTD
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/dtd -d DTD --seed 0 --finetune --log logs/co_tuning/dtd/1e-2 --lr 1e-2
# caltech101
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/caltech101 -d Caltech101 --seed 0 --finetune --log logs/co_tuning/caltech101/lr_1e-3 --lr 1e-3
# SUN397
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/sun397 -d SUN397 --seed 0 --finetune --log logs/co_tuning/sun397/lr_1e-2 --lr 1e-2
# Food 101
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/food-101 -d Food101 --seed 0 --finetune --log logs/co_tuning/food-101/lr_1e-2 --lr 1e-2
# Standford Cars
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/stanford_cars -d StanfordCars --seed 0 --finetune --log logs/co_tuning/stanford_cars/lr_1e-2 --lr 1e-2
# Standford Cars
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/aircraft -d Aircraft --seed 0 --finetune --log logs/co_tuning/aircraft/lr_1e-2 --lr 1e-2


# MoCo (Unsupervised Pretraining)
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cub200 -d CUB200 -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/cub200_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cub200 -d CUB200 -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/cub200_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/cub200 -d CUB200 -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/cub200_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/cars_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/cars_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/cars_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/cars_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/aircraft -d Aircraft -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/aircraft_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/aircraft -d Aircraft -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/aircraft_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/aircraft -d Aircraft -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/aircraft_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python co_tuning.py data/aircraft -d Aircraft -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning/aircraft_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth
