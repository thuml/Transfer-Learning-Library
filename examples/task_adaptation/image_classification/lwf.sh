#!/usr/bin/env bash
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/lwf/cub200_100 --lr 0.01
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log logs/lwf/cub200_50 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log logs/lwf/cub200_30 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log logs/lwf/cub200_15 --lr 0.001

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python lwf.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --finetune --log logs/lwf/car_100 --lr 0.01
CUDA_VISIBLE_DEVICES=0 python lwf.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --finetune --log logs/lwf/car_50 --lr 0.01
CUDA_VISIBLE_DEVICES=0 python lwf.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --finetune --log logs/lwf/car_30 --lr 0.01
CUDA_VISIBLE_DEVICES=0 python lwf.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --finetune --log logs/lwf/car_15 --lr 0.01

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python lwf.py data/aircraft -d Aircraft -sr 100 --seed 0 --finetune --log logs/lwf/aircraft_100 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/aircraft -d Aircraft -sr 50 --seed 0 --finetune --log logs/lwf/aircraft_50 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/aircraft -d Aircraft -sr 30 --seed 0 --finetune --log logs/lwf/aircraft_30 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/aircraft -d Aircraft -sr 15 --seed 0 --finetune --log logs/lwf/aircraft_15 --lr 0.001

# CIFAR10
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cifar10 -d CIFAR10 --seed 0 --finetune --log logs/lwf/cifar10/1e-2 --lr 1e-2
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cifar100 -d CIFAR100 --seed 0 --finetune --log logs/lwf/cifar100/1e-2 --lr 1e-2
# Flowers
CUDA_VISIBLE_DEVICES=0 python lwf.py data/oxford_flowers102 -d OxfordFlowers102 --seed 0 --finetune --log logs/lwf/oxford_flowers102/1e-2 --lr 1e-2
# Pets
CUDA_VISIBLE_DEVICES=0 python lwf.py data/oxford_pet -d OxfordIIITPets --seed 0 --finetune --log logs/lwf/oxford_pet/1e-2 --lr 1e-2
# DTD
CUDA_VISIBLE_DEVICES=0 python lwf.py data/dtd -d DTD --seed 0 --finetune --log logs/lwf/dtd/1e-2 --lr 1e-2
# caltech101
CUDA_VISIBLE_DEVICES=0 python lwf.py data/caltech101 -d Caltech101 --seed 0 --finetune --log logs/lwf/caltech101/lr_1e-3 --lr 1e-3
# SUN397
CUDA_VISIBLE_DEVICES=0 python lwf.py data/sun397 -d SUN397 --seed 0 --finetune --log logs/lwf/sun397/lr_1e-2 --lr 1e-2
# Food 101
CUDA_VISIBLE_DEVICES=0 python lwf.py data/food-101 -d Food101 --seed 0 --finetune --log logs/lwf/food-101/lr_1e-2 --lr 1e-2
# Standford Cars
CUDA_VISIBLE_DEVICES=0 python lwf.py data/stanford_cars -d StanfordCars --seed 0 --finetune --log logs/lwf/stanford_cars/lr_1e-2 --lr 1e-2
# Standford Cars
CUDA_VISIBLE_DEVICES=0 python lwf.py data/aircraft -d Aircraft --seed 0 --finetune --log logs/lwf/aircraft/lr_1e-2 --lr 1e-2
