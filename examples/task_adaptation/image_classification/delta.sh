#!/usr/bin/env bash
# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/delta/cub200_100
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log logs/delta/cub200_50
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log logs/delta/cub200_30
CUDA_VISIBLE_DEVICES=0 python delta.py data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log logs/delta/cub200_15

# Stanford Cars
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --finetune --log logs/delta/car_100
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --finetune --log logs/delta/car_50
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --finetune --log logs/delta/car_30
CUDA_VISIBLE_DEVICES=0 python delta.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --finetune --log logs/delta/car_15

# Aircrafts
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 100 --seed 0 --finetune --log logs/delta/aircraft_100
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 50 --seed 0 --finetune --log logs/delta/aircraft_50
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 30 --seed 0 --finetune --log logs/delta/aircraft_30
CUDA_VISIBLE_DEVICES=0 python delta.py data/aircraft -d Aircraft -sr 15 --seed 0 --finetune --log logs/delta/aircraft_15

# Resisc45
CUDA_VISIBLE_DEVICES=0 python delta.py data/vtab/resisc45 -d Resisc45 -sc 10 --finetune --seed 0 \
  --log logs/delta/resisc45_10
CUDA_VISIBLE_DEVICES=0 python delta.py data/vtab/resisc45 -d Resisc45 -sc 20 --finetune --seed 0 \
  --log logs/delta/resisc45_20
CUDA_VISIBLE_DEVICES=0 python delta.py data/vtab/resisc45 -d Resisc45 -sc 40 --finetune --seed 0 \
  --log logs/delta/resisc45_40
CUDA_VISIBLE_DEVICES=0 python delta.py data/vtab/resisc45 -d Resisc45 -sc 80 --finetune --seed 0 \
  --log logs/delta/resisc45_80

# Patch Camelyon
CUDA_VISIBLE_DEVICES=0 python delta.py data/vtab/patch_camelyon -d PatchCamelyon -sc 40 --finetune \
  --seed 0 --log logs/delta/patch_camelyon_40
CUDA_VISIBLE_DEVICES=0 python delta.py data/vtab/patch_camelyon -d PatchCamelyon -sc 80 --finetune \
  --seed 0 --log logs/delta/patch_camelyon_80
CUDA_VISIBLE_DEVICES=0 python delta.py data/vtab/patch_camelyon -d PatchCamelyon -sc 160 --finetune \
  --seed 0 --log logs/delta/patch_camelyon_160
CUDA_VISIBLE_DEVICES=0 python delta.py data/vtab/patch_camelyon -d PatchCamelyon -sc 320 --finetune \
  --seed 0 --log logs/delta/patch_camelyon_320

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
