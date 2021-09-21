#!/usr/bin/env bash
# Supervised Pretraining
# CUB-200-2011
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/baseline/cub200_100
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log logs/baseline/cub200_50
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log logs/baseline/cub200_30
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log logs/baseline/cub200_15

# Standford Cars
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --finetune --log logs/baseline/car_100
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --finetune --log logs/baseline/car_50
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --finetune --log logs/baseline/car_30
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --finetune --log logs/baseline/car_15

# Aircrafts
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 100 --seed 0 --finetune --log logs/baseline/aircraft_100
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 50 --seed 0 --finetune --log logs/baseline/aircraft_50
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 30 --seed 0 --finetune --log logs/baseline/aircraft_30
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 15 --seed 0 --finetune --log logs/baseline/aircraft_15

# MoCo (Unsupervised Pretraining)
#CUB-200-2011
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cub200_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cub200_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/cub200 -d CUB200 -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cub200_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Standford Cars
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cars_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cars_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cars_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/cars_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Aircrafts
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 100 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/aircraft_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 50 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/aircraft_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 30 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/aircraft_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/aircraft -d Aircraft -sr 15 --seed 0 --lr 0.1 --finetune -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_baseline/aircraft_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth

# Supervised Pretraining
# Finetune on VTAB
# Natural
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/caltech101 -d Caltech101 -ss 1000 --seed 0 --log logs/baseline/vtab/caltech101
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/cifar100 -d Cifar100 -ss 1000 --seed 0 --log logs/baseline/vtab/cifar100
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/dtd -d DTD -ss 1000 --seed 0 --log logs/baseline/vtab/dtd
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/oxford_flowers102 -d Flowers102 -ss 1000 --seed 0 --log logs/baseline/vtab/oxford_flowers102
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/oxford_iiit_pet -d Pets -ss 1000 --seed 0 --log logs/baseline/vtab/oxford_iiit_pet
# CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/sun397 -d Sun397 -ss 1000 --seed 0 --log logs/baseline/vtab/sun397
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/svhn_cropped -d SVHN -ss 1000 --seed 0 --log logs/baseline/vtab/svhn_cropped

# Specialized
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/eurosat -d EuroSAT -ss 1000 --seed 0 --log logs/baseline/vtab/eurosat
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/patch_camelyon -d PatchCamelyon -ss 1000 --seed 0 --log logs/baseline/vtab/patch_camelyon

# Structured
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/smallnorb_azimuth -d SmallnorbAzimuth -ss 1000 --seed 0 --log logs/baseline/vtab/smallnorb_azimuth
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/smallnorb_elevation -d SmallnorblElevation -ss 1000 --seed 0 --log logs/baseline/vtab/smallnorb_elevation
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/dsprites_loc -d DSpritesLocation -ss 1000 --seed 0 --log logs/baseline/vtab/dsprites_loc
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/dsprites_distance -d DSpritesLocation -ss 1000 --seed 0 --log logs/baseline/vtab/dsprites_distance
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/kitti_distance -d KITTIDist -ss 1000 --seed 0 --log logs/baseline/vtab/kitti_distance
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/clevr_count -d ClevrCount -ss 1000 --seed 0 --log logs/baseline/vtab/clevr_count
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/clevr_distance -d ClevrDistance -ss 1000 --seed 0 --log logs/baseline/vtab/clevr_distance
 CUDA_VISIBLE_DEVICES=0 python baseline.py data/vtab/dmlab -d DMLab -ss 1000 --seed 0 --log logs/baseline/vtab/dmlab
