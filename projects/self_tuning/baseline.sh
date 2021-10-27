#!/usr/bin/env bash
# Supervised Pretraining
# ResNet50, CUB200
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/baseline/cub200_15
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 30 --seed 0 --log logs/baseline/cub200_30
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 50 --seed 0 --log logs/baseline/cub200_50

# ResNet50, StanfordCars
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/baseline/car_15
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/baseline/car_30
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/baseline/car_50

# ResNet50, Aircraft
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/baseline/aircraft_15
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/baseline/aircraft_30
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/baseline/aircraft_50

# MoCo (Unsupervised Pretraining)
# ResNet50, CUB200
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 15 --seed 0 --log logs/baseline_moco/cub200_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 30 --seed 0 --log logs/baseline_moco/cub200_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 50 --seed 0 --log logs/baseline_moco/cub200_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# ResNet50, StanfordCars
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 15 --seed 0 --log logs/baseline_moco/car_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 30 --seed 0 --log logs/baseline_moco/car_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 50 --seed 0 --log logs/baseline_moco/car_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# ResNet50, Aircraft
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 15 --seed 0 --log logs/baseline_moco/aircraft_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 30 --seed 0 --log logs/baseline_moco/aircraft_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft --lr 0.1 --epochs 12 --milestones 3 6 9 \
  -i 2000 -sr 50 --seed 0 --log logs/baseline_moco/aircraft_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
