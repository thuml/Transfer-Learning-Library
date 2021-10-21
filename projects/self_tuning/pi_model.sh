#!/usr/bin/env bash
# ResNet50, CUB200
CUDA_VISIBLE_DEVICES=0,1 python pi_model.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/pi_model/cub200_15
CUDA_VISIBLE_DEVICES=0,1 python pi_model.py data/cub200 -d CUB200 -sr 30 --seed 0 --log logs/pi_model/cub200_30
CUDA_VISIBLE_DEVICES=0,1 python pi_model.py data/cub200 -d CUB200 -sr 50 --seed 0 --log logs/pi_model/cub200_50

# ResNet50, StanfordCars
CUDA_VISIBLE_DEVICES=0,1 python pi_model.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/pi_model/car_15
CUDA_VISIBLE_DEVICES=0,1 python pi_model.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/pi_model/car_30
CUDA_VISIBLE_DEVICES=0,1 python pi_model.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/pi_model/car_50

# ResNet50, Aircraft
CUDA_VISIBLE_DEVICES=0,1 python pi_model.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/pi_model/aircraft_15
CUDA_VISIBLE_DEVICES=0,1 python pi_model.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/pi_model/aircraft_30
CUDA_VISIBLE_DEVICES=0,1 python pi_model.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/pi_model/aircraft_50
