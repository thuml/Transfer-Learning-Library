#!/usr/bin/env bash
#cub200
CUDA_VISIBLE_DEVICES=4 python StochNorm.py data/cub200 -d  CUB200 -a resnet50 -sr 100 --epochs 100 --seed 0 --log logs/StochNorm/CUB200_100

CUDA_VISIBLE_DEVICES=4 python StochNorm.py data/cub200 -d  CUB200  -a resnet50 -sr 50 --epochs 100 --seed 0 --log logs/StochNorm/CUB200_50

CUDA_VISIBLE_DEVICES=5 python StochNorm.py data/cub200 -d  CUB200  -a resnet50 -sr 30 --epochs 100 --seed 0 --log logs/StochNorm/CUB200_30

CUDA_VISIBLE_DEVICES=5 python StochNorm.py data/cub200 -d  CUB200  -a resnet50 -sr 15 --epochs 100 --seed 0 --log logs/StochNorm/CUB200_15
# car

CUDA_VISIBLE_DEVICES=5 python StochNorm.py data/stanford_cars  --lr 0.01 -d stanford_cars -a resnet50 -sr 100 --epochs 100 --seed 0 --log logs/StochNorm/car_100

CUDA_VISIBLE_DEVICES=5 python StochNorm.py data/stanford_cars  --lr 0.01 -d stanford_cars -a resnet50 -sr 50 --epochs 100 --seed 0 --log logs/StochNorm/car_50

CUDA_VISIBLE_DEVICES=4 python StochNorm.py data/stanford_cars  --lr 0.01 -d stanford_cars -a resnet50 -sr 30 --epochs 100 --seed 0 --log logs/StochNorm/car_30

CUDA_VISIBLE_DEVICES=4 python StochNorm.py data/stanford_cars  --lr 0.01 -d stanford_cars -a resnet50 -sr 15 --epochs 100 --seed 0 --log logs/StochNorm/car_15

