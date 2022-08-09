#!/usr/bin/env bash
# Supervised Pretraining
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/bss/cub200_100
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log logs/bss/cub200_50
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log logs/bss/cub200_30
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log logs/bss/cub200_15

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --finetune --log logs/bss/car_100
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --finetune --log logs/bss/car_50
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --finetune --log logs/bss/car_30
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --finetune --log logs/bss/car_15

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 100 --seed 0 --finetune --log logs/bss/aircraft_100
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 50 --seed 0 --finetune --log logs/bss/aircraft_50
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 30 --seed 0 --finetune --log logs/bss/aircraft_30
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 15 --seed 0 --finetune --log logs/bss/aircraft_15

# CIFAR10
CUDA_VISIBLE_DEVICES=0 python bss.py data/cifar10 -d CIFAR10 --seed 0 --finetune --log logs/bss/cifar10/1e-2 --lr 1e-2
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python bss.py data/cifar100 -d CIFAR100 --seed 0 --finetune --log logs/bss/cifar100/1e-2 --lr 1e-2
# Flowers
CUDA_VISIBLE_DEVICES=0 python bss.py data/oxford_flowers102 -d OxfordFlowers102 --seed 0 --finetune --log logs/bss/oxford_flowers102/1e-2 --lr 1e-2
# Pets
CUDA_VISIBLE_DEVICES=0 python bss.py data/oxford_pet -d OxfordIIITPets --seed 0 --finetune --log logs/bss/oxford_pet/1e-2 --lr 1e-2
# DTD
CUDA_VISIBLE_DEVICES=0 python bss.py data/dtd -d DTD --seed 0 --finetune --log logs/bss/dtd/1e-2 --lr 1e-2
# caltech101
CUDA_VISIBLE_DEVICES=0 python bss.py data/caltech101 -d Caltech101 --seed 0 --finetune --log logs/bss/caltech101/lr_1e-3 --lr 1e-3
# SUN397
CUDA_VISIBLE_DEVICES=0 python bss.py data/sun397 -d SUN397 --seed 0 --finetune --log logs/bss/sun397/lr_1e-2 --lr 1e-2
# Food 101
CUDA_VISIBLE_DEVICES=0 python bss.py data/food-101 -d Food101 --seed 0 --finetune --log logs/bss/food-101/lr_1e-2 --lr 1e-2
# Standford Cars
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars --seed 0 --finetune --log logs/bss/stanford_cars/lr_1e-2 --lr 1e-2
# Standford Cars
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft --seed 0 --finetune --log logs/bss/aircraft/lr_1e-2 --lr 1e-2

# MoCo (Unsupervised Pretraining)
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/cub200_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/cub200_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/cub200_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/cars_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/cars_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/cars_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/cars_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/aircraft_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/aircraft_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/aircraft_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bss/aircraft_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth
