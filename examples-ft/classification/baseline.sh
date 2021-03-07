#!/usr/bin/env bash
#cub200
CUDA_VISIBLE_DEVICES=4 python baseline.py data/cub200 -d  CUB200 -a resnet50 -sr 100 --epochs 100 --seed 0 --log logs/baseline/CUB200_100

CUDA_VISIBLE_DEVICES=4 python baseline.py data/cub200 -d  CUB200  -a resnet50 -sr 50 --epochs 100 --seed 0 --log logs/baseline/CUB200_50

CUDA_VISIBLE_DEVICES=5 python baseline.py data/cub200 -d  CUB200  -a resnet50 -sr 30 --epochs 100 --seed 0 --log logs/baseline/CUB200_30

CUDA_VISIBLE_DEVICES=5 python baseline.py data/cub200 -d  CUB200  -a resnet50 -sr 15 --epochs 100 --seed 0 --log logs/baseline/CUB200_15


# car
CUDA_VISIBLE_DEVICES=5 python baseline.py data/stanford_cars --lr 0.01 -d  stanford_cars -a resnet50 -sr 15 --epochs 100 --seed 0 --log logs/baseline/car_15_lr_0_01

CUDA_VISIBLE_DEVICES=5 python baseline.py data/stanford_cars --lr 0.01 -d  stanford_cars -a resnet50 -sr 30 --epochs 100 --seed 0 --log logs/baseline/car_30_lr_0_01

CUDA_VISIBLE_DEVICES=4 python baseline.py data/stanford_cars --lr 0.01 -d  stanford_cars -a resnet50 -sr 50 --epochs 100 --seed 0 --log logs/baseline/car_50_lr_0_01

CUDA_VISIBLE_DEVICES=4 python baseline.py data/stanford_cars --lr 0.01 -d  stanford_cars -a resnet50 -sr 100 --epochs 100 --seed 0 --log logs/baseline/car_100_lr_0_01

