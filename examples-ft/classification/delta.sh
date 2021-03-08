#!/usr/bin/env bash
#CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 --reg_type fea_map --alpha 0.0001 --lr 0.01 -d CUB200 -sr 100 --seed 0 --log logs/delta/cub200_100_30epochs
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 --reg_type fea_map --alpha 0.0001 --lr 0.01 -d CUB200 -sr 50 --seed 0 --log logs/delta/cub200_50_30epochs
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 --reg_type fea_map --alpha 0.0001 --lr 0.01 -d CUB200 -sr 30 --seed 0 --log logs/delta/cub200_30_30epochs
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 --reg_type fea_map --alpha 0.0001 --lr 0.01 -d CUB200 -sr 15 --seed 0 --log logs/delta/cub200_15_30epochs

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars --lr 0.01 -d StanfordCars -sr 100 --seed 0 --log logs/baseline/car_100
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars --lr 0.01 -d StanfordCars -sr 50 --seed 0 --log logs/baseline/car_50
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars --lr 0.01 -d StanfordCars -sr 30 --seed 0 --log logs/baseline/car_30
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars --lr 0.01 -d StanfordCars -sr 15 --seed 0 --log logs/baseline/car_15
