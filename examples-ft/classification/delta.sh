#!/usr/bin/env bash
#CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 --alpha 0.0001 -d CUB200 -sr 100 --seed 0 --log logs/delta/cub200_100
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 --alpha 0.0001 -d CUB200 -sr 50 --seed 0 --log logs/delta/cub200_50
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 --alpha 0.0001 -d CUB200 -sr 30 --seed 0 --log logs/delta/cub200_30
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 --alpha 0.0001 -d CUB200 -sr 15 --seed 0 --log logs/delta/cub200_15

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --log logs/delta/car_100
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/delta/car_50
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/delta/car_30
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/delta/car_15

# aircrafts
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 100 --seed 0 --log logs/delta/aircraft_100
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/delta/aircraft_50
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/delta/aircraft_30
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/delta/aircraft_15