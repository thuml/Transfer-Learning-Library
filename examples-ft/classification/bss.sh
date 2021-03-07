#!/usr/bin/env bash
#cub200
CUDA_VISIBLE_DEVICES=4 python BSS.py data/cub200 -d  CUB200 -a resnet50 -sr 100  -k 1 -t 0.001 --epochs 100 --seed 0 --log logs/BSS/CUB200_100

CUDA_VISIBLE_DEVICES=5 python BSS.py data/cub200 -d  CUB200  -a resnet50 -sr 50 -k 1 -t 0.001 --epochs 100 --seed 0 --log logs/BSS/CUB200_50

CUDA_VISIBLE_DEVICES=4 python BSS.py data/cub200 -d  CUB200  -a resnet50 -sr 15  -k 1 -t 0.001 --epochs 100 --seed 0 --log logs/BSS/CUB200_15

CUDA_VISIBLE_DEVICES=5 python BSS.py data/cub200 -d  CUB200  -a resnet50 -sr 30 -k 1 -t 0.001  --epochs 100 --seed 0 --log logs/BSS/CUB200_30


# car
CUDA_VISIBLE_DEVICES=4 python BSS.py data/car --lr 0.01 -d  stanford_cars -a resnet50 -sr 100 --epochs 100 --seed 0 --log logs/baseline/car_100

CUDA_VISIBLE_DEVICES=5 python BSS.py data/car --lr 0.01 -d  stanford_cars -a resnet50 -sr 50 --epochs 100 --seed 0 --log logs/baseline/car_50_100

CUDA_VISIBLE_DEVICES=4 python BSS.py data/car --lr 0.01 -d  stanford_cars -a resnet50 -sr 30 --epochs 100 --seed 0 --log logs/baseline/car_30_100

CUDA_VISIBLE_DEVICES=5 python BSS.py data/car --lr 0.01 -d  stanford_cars -a resnet50 -sr 15 --epochs 100 --seed 0 --log logs/baseline/car_15_100