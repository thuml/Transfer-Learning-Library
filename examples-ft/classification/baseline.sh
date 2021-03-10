#!/usr/bin/env bash
#CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 --lr 0.01 -d CUB200 -sr 100 --seed 0 --log logs/baseline/cub200_100_30epochs_6_20
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 --lr 0.01 -d CUB200 -sr 50 --seed 0 --log logs/baseline/cub200_50_30epochs_6_20
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 --lr 0.01 -d CUB200 -sr 30 --seed 0 --log logs/baseline/cub200_30_30epochs_6_20
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 --lr 0.01 -d CUB200 -sr 15 --seed 0 --log logs/baseline/cub200_15_30epochs_6_20

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars --lr-decay-epochs 12 20 -d StanfordCars -sr 100 --seed 0 --log logs/baseline/car_100
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars --lr-decay-epochs 12 20 --lr 0.01 -d StanfordCars -sr 50 --seed 0 --log logs/baseline/car_50
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars --lr-decay-epochs 12 20 --lr 0.01 -d StanfordCars -sr 30 --seed 0 --log logs/baseline/car_30
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars --lr-decay-epochs 12 20 --lr 0.01 -d StanfordCars -sr 15 --seed 0 --log logs/baseline/car_15

# aircrafts
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 100 --seed 0 --log logs/baseline/aircraft_100
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/baseline/aircraft_50
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/baseline/aircraft_30
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/baseline/aircraft_15

