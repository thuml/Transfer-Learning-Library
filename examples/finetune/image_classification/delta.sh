#!/usr/bin/env bash
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 100 --seed 0 --log logs/delta/cub200_100
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 50 --seed 0 --log logs/delta/cub200_50
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 30 --seed 0 --log logs/delta/cub200_30
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/delta/cub200_15

# Stanford Cars
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --log logs/delta/car_100 
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/delta/car_50 
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/delta/car_30 
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/delta/car_15 

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 100 --seed 0 --log logs/delta/aircraft_100
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/delta/aircraft_50
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/delta/aircraft_30
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/delta/aircraft_15

# Stanford Dogs
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_dogs -d StanfordDogs -sr 100 --seed 0 --log logs/delta/dogs_100 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_dogs -d StanfordDogs -sr 50 --seed 0 --log logs/delta/dogs_50 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_dogs -d StanfordDogs -sr 30 --seed 0 --log logs/delta/dogs_30 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_dogs -d StanfordDogs -sr 15 --seed 0 --log logs/delta/dogs_15 --lr 0.001

# Oxford-III pet
CUDA_VISIBLE_DEVICES=0 python delta.py data/oxford_pet -d OxfordIIITPet -sr 100 --seed 0 --log logs/delta/pet_100 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python delta.py data/oxford_pet -d OxfordIIITPet -sr 50 --seed 0 --log logs/delta/pet_50 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python delta.py data/oxford_pet -d OxfordIIITPet -sr 30 --seed 0 --log logs/delta/pet_30 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python delta.py data/oxford_pet -d OxfordIIITPet -sr 15 --seed 0 --log logs/delta/pet_15 --lr 0.001

# COCO-70
CUDA_VISIBLE_DEVICES=0 python delta.py data/coco70 -d COCO70 -sr 100 --seed 0 --log logs/delta/coco_100
CUDA_VISIBLE_DEVICES=0 python delta.py data/coco70 -d COCO70 -sr 50 --seed 0 --log logs/delta/coco_50
CUDA_VISIBLE_DEVICES=0 python delta.py data/coco70 -d COCO70 -sr 30 --seed 0 --log logs/delta/coco_30
CUDA_VISIBLE_DEVICES=0 python delta.py data/coco70 -d COCO70 -sr 15 --seed 0 --log logs/delta/coco_15

# MoCo (Unsupervised Pretraining)
#CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/cub200_100 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 50 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/cub200_50 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 30 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/cub200_30 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 15 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/cub200_15 --pretrained checkpoints/moco_v1_200ep_pretrain.pth

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/cars_100 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/cars_50 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/cars_30 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/cars_15 --pretrained checkpoints/moco_v1_200ep_pretrain.pth

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/aircraft_100 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 50 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/aircraft_50 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 30 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/aircraft_30 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 15 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_delta/aircraft_15 --pretrained checkpoints/moco_v1_200ep_pretrain.pth
