#!/usr/bin/env bash
# Supervised Pretraining
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 100 --hflip --seed 0 --log logs/baseline/cub200_100
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 50 --hflip --seed 0 --log logs/baseline/cub200_50
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 30 --hflip --seed 0 --log logs/baseline/cub200_30
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 15 --hflip --seed 0 --log logs/baseline/cub200_15

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 100 --hflip --seed 0 --log logs/baseline/car_100
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 50 --hflip --seed 0 --log logs/baseline/car_50
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 30 --hflip --seed 0 --log logs/baseline/car_30
CUDA_VISIBLE_DEVICES=7 python baseline.py data/stanford_cars -d StanfordCars -sr 15 --hflip --seed 0 --log logs/baseline/car_15

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 100 --hflip --seed 0 --log logs/baseline/aircraft_100
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 50 --hflip --seed 0 --log logs/baseline/aircraft_50
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 30 --hflip --seed 0 --log logs/baseline/aircraft_30
CUDA_VISIBLE_DEVICES=6 python baseline.py data/aircraft -d Aircraft -sr 15 --hflip --seed 0 --log logs/baseline/aircraft_15

# Stanford Dogs
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_dogs -d StanfordDogs -sr 100 --hflip --seed 0 --log logs/baseline/dogs_100 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_dogs -d StanfordDogs -sr 50 --hflip --seed 0 --log logs/baseline/dogs_50 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_dogs -d StanfordDogs -sr 30 --hflip --seed 0 --log logs/baseline/dogs_30 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_dogs -d StanfordDogs -sr 15 --hflip --seed 0 --log logs/baseline/dogs_15 --lr 0.001

# Oxford-III pet
CUDA_VISIBLE_DEVICES=0 python baseline.py data/oxford_pet -d OxfordIIITPet -sr 100 --hflip --seed 0 --log logs/baseline/pet_100 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python baseline.py data/oxford_pet -d OxfordIIITPet -sr 50 --hflip --seed 0 --log logs/baseline/pet_50 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python baseline.py data/oxford_pet -d OxfordIIITPet -sr 30 --hflip --seed 0 --log logs/baseline/pet_30 --lr 0.001
CUDA_VISIBLE_DEVICES=0 python baseline.py data/oxford_pet -d OxfordIIITPet -sr 15 --hflip --seed 0 --log logs/baseline/pet_15 --lr 0.001

# COCO-70
CUDA_VISIBLE_DEVICES=0 python baseline.py data/coco70 -d COCO70 -sr 100 --hflip --seed 0 --log logs/baseline/coco_100
CUDA_VISIBLE_DEVICES=0 python baseline.py data/coco70 -d COCO70 -sr 50 --hflip --seed 0 --log logs/baseline/coco_50
CUDA_VISIBLE_DEVICES=0 python baseline.py data/coco70 -d COCO70 -sr 30 --hflip --seed 0 --log logs/baseline/coco_30
CUDA_VISIBLE_DEVICES=0 python baseline.py data/coco70 -d COCO70 -sr 15 --hflip --seed 0 --log logs/baseline/coco_15

# MoCo (Unsupervised Pretraining)
#CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 100 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 50 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cub200_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 30 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cub200_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 15 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cub200_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Standford Cars
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 100 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cars_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 50 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cars_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 30 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cars_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 15 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cars_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 100 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/aircraft_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 50 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/aircraft_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 30 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/aircraft_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 15 --hflip --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/aircraft_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Supervised Pretraining
# Finetune on VTAB
# Natural
CUDA_VISIBLE_DEVICES=6 python baseline.py data/vtab/caltech101 -d caltech101 -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/caltech101
CUDA_VISIBLE_DEVICES=7 python baseline.py data/vtab/cifar100 -d cifar100 -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/cifar100
CUDA_VISIBLE_DEVICES=7 python baseline.py data/vtab/dtd -d dtd -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/dtd
CUDA_VISIBLE_DEVICES=7 python baseline.py data/vtab/oxford_flowers102 -d oxford_flowers102 -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/oxford_flowers102
CUDA_VISIBLE_DEVICES=4 python baseline.py data/vtab/oxford_iiit_pet -d oxford_iiit_pet -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/oxford_iiit_pet
#CUDA_VISIBLE_DEVICES=7 python baseline.py data/vtab/sun397 -d sun397 -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/sun397
#CUDA_VISIBLE_DEVICES=7 python baseline.py data/vtab/svhn_cropped -d svhn_cropped -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/svhn_cropped

# Specialized
CUDA_VISIBLE_DEVICES=7 python baseline.py data/vtab/patch_camelyon -d patch_camelyon -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/patch_camelyon

# Structured
CUDA_VISIBLE_DEVICES=7 python baseline.py data/vtab/smallnorb_azimuth -d smallnorb_azimuth -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/smallnorb_azimuth
CUDA_VISIBLE_DEVICES=7 python baseline.py data/vtab/smallnorb_elevation -d smallnorb_elevation -ss 1000 --hflip --seed 0 --log logs/baseline/vtab/smallnorb_elevation




