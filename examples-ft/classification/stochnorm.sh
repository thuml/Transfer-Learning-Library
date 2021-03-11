#!/usr/bin/env bash
#CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/cub200 -d CUB200 -sr 100 --seed 0 --log logs/stochnorm/cub200_100
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/cub200 -d CUB200 -sr 50 --seed 0 --log logs/stochnorm/cub200_50
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/cub200 -d CUB200 -sr 30 --seed 0 --log logs/stochnorm/cub200_30
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/stochnorm/cub200_15

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --log logs/stochnorm/car_100
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/stochnorm/car_50
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/stochnorm/car_30
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/stochnorm/car_15

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/aircraft -d Aircraft -sr 100 --seed 0 --log logs/stochnorm/aircraft_100
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/stochnorm/aircraft_50
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/stochnorm/aircraft_30
CUDA_VISIBLE_DEVICES=0 python stochnorm.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/stochnorm/aircraft_15
