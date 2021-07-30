#!/usr/bin/env bash
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cub200 -d CUB200 -sr 100 --seed 0 --log logs/lwf/cub200_100 --lr 0.01
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cub200 -d CUB200 -sr 50 --seed 0 --log logs/lwf/cub200_50 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cub200 -d CUB200 -sr 30 --seed 0 --log logs/lwf/cub200_30 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/lwf/cub200_15 --lr 0.001

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python lwf.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --log logs/lwf/car_100 --lr 0.01
CUDA_VISIBLE_DEVICES=0 python lwf.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/lwf/car_50 --lr 0.01
CUDA_VISIBLE_DEVICES=0 python lwf.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/lwf/car_30 --lr 0.01
CUDA_VISIBLE_DEVICES=0 python lwf.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/lwf/car_15 --lr 0.01

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python lwf.py data/aircraft -d Aircraft -sr 100 --seed 0 --log logs/lwf/aircraft_100 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/lwf/aircraft_50 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/lwf/aircraft_30 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python lwf.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/lwf/aircraft_15 --lr 0.001