#!/usr/bin/env bash
#CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 100 --seed 0 --log logs/bss/cub200_100
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 50 --seed 0 --log logs/bss/cub200_50
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 30 --seed 0 --log logs/bss/cub200_30
CUDA_VISIBLE_DEVICES=0 python bss.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/bss/cub200_15

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --log logs/bss/car_100
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/bss/car_50
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/bss/car_30
CUDA_VISIBLE_DEVICES=0 python bss.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/bss/car_15

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 100 --seed 0 --log logs/bss/aircraft_100
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/bss/aircraft_50
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/bss/aircraft_30
CUDA_VISIBLE_DEVICES=0 python bss.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/bss/aircraft_15
