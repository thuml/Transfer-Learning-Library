#!/usr/bin/env bash
# Supervised Pretraining
# ResNet50, CUB200
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/cub200 -d CUB200 -sr 15 --seed 0 --log logs/fix_match/cub200_15
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/cub200 -d CUB200 -sr 30 --seed 0 --log logs/fix_match/cub200_30
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/cub200 -d CUB200 -sr 50 --seed 0 --log logs/fix_match/cub200_50

# ResNet50, StanfordCars
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --log logs/fix_match/car_15
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --log logs/fix_match/car_30
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --log logs/fix_match/car_50

# ResNet50, Aircraft
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/aircraft -d Aircraft -sr 15 --seed 0 --log logs/fix_match/aircraft_15
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/aircraft -d Aircraft -sr 30 --seed 0 --log logs/fix_match/aircraft_30
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/aircraft -d Aircraft -sr 50 --seed 0 --log logs/fix_match/aircraft_50

# MoCo (Unsupervised Pretraining)
# ResNet50, CUB200
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/cub200 -d CUB200 -i 2000 -sr 15 --seed 0 --log logs/fix_match_moco/cub200_15
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/cub200 -d CUB200 -i 2000 -sr 30 --seed 0 --log logs/fix_match_moco/cub200_30
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/cub200 -d CUB200 -i 2000 -sr 50 --seed 0 --log logs/fix_match_moco/cub200_50

# ResNet50, StanfordCars
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/stanford_cars -d StanfordCars -i 2000 -sr 15 --seed 0 --log logs/fix_match_moco/car_15 \
  --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/stanford_cars -d StanfordCars -i 2000 -sr 30 --seed 0 --log logs/fix_match_moco/car_30 \
  --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/stanford_cars -d StanfordCars -i 2000 -sr 50 --seed 0 --log logs/fix_match_moco/car_50 \
  --pretrained checkpoints/moco_v1_200ep_backbone.pth

# ResNet50, Aircraft
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/aircraft -d Aircraft -i 2000 -sr 15 --seed 0 --log logs/fix_match_moco/aircraft_15 \
  --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/aircraft -d Aircraft -i 2000 -sr 30 --seed 0 --log logs/fix_match_moco/aircraft_30 \
  --pretrained checkpoints/moco_v1_200ep_backbone.pth
CUDA_VISIBLE_DEVICES=0,1 python fix_match.py data/aircraft -d Aircraft -i 2000 -sr 50 --seed 0 --log logs/fix_match_moco/aircraft_50 \
  --pretrained checkpoints/moco_v1_200ep_backbone.pth
