#!/usr/bin/env bash
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 100 --seed 0 --log logs/delta/cub200_100 --channel-weight weight-cub.json
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 50 --seed 0 --log logs/delta/cub200_50 --channel-weight weight-cub.json
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 30 --seed 0 --log logs/delta/cub200_30 --channel-weight weight-cub.json
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/delta/cub200_15 --channel-weight weight-cub.json

# Stanford Cars
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --log logs/delta/car_100 --channel-weight weight-car.json
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/delta/car_50 --channel-weight weight-car.json
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/delta/car_30 --channel-weight weight-car.json
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/delta/car_15 --channel-weight weight-car.json

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 100 --seed 0 --log logs/delta/aircraft_100 --channel-weight weight-aircraft.json
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/delta/aircraft_50 --channel-weight weight-aircraft.json
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/delta/aircraft_30 --channel-weight weight-aircraft.json
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/delta/aircraft_15 --channel-weight weight-aircraft.json

