#!/usr/bin/env bash
# ResNet50, Office31, Single Source
CUDA_VISIBLE_DEVICES=0 python afn.py data/office31 -d Office31 -s A -t W -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python afn.py data/office31 -d Office31 -s D -t W -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_D2W
CUDA_VISIBLE_DEVICES=0 python afn.py data/office31 -d Office31 -s W -t D -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_W2D
CUDA_VISIBLE_DEVICES=0 python afn.py data/office31 -d Office31 -s A -t D -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_A2D
CUDA_VISIBLE_DEVICES=0 python afn.py data/office31 -d Office31 -s D -t A -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_D2A
CUDA_VISIBLE_DEVICES=0 python afn.py data/office31 -d Office31 -s W -t A -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_W2A

# ResNet50, Office-Home, Single Source
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Rw2Pr

# ResNet101, VisDA-2017, Single Source
CUDA_VISIBLE_DEVICES=0 python afn.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 -r 0.3 -b 36 \
    --epochs 10 -i 1000 --seed 0 --per-class-eval --center-crop --log logs/afn/VisDA2017

# ResNet50, Wilds Dataset
CUDA_VISIBLE_DEVICES=0 python afn.py data/wilds -d iwildcam -a resnet50 --epochs 30 -i 1000 --seed 0 --log logs/afn/iwildcam
CUDA_VISIBLE_DEVICES=0 python afn.py data/wilds -d fmow -a resnet50 --epochs 30 -i 1000 --seed 0 --log logs/afn/fmow

# ResNet50, ImageNet200 -> ImageNetR
CUDA_VISIBLE_DEVICES=0 python afn.py data/ImageNetR -d ImageNetR -s IN -t INR -a resnet50 --epochs 20 -i 2500 --seed 0 --log logs/afn/ImageNet_IN2INR

# ig_resnext101_32x8d, ImageNet -> ImageNetSketch
CUDA_VISIBLE_DEVICES=0 python afn.py data/imagenet-sketch -d ImageNetSketch -s IN -t sketch -a ig_resnext101_32x8d --epochs 20 -i 2500 --seed 0 --log logs/afn_ig_resnext101_32x8d/ImageNet_IN2sketch

# Vision Transformer, Office-Home, Single Source
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Ar -t Cl -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Ar -t Pr -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Ar -t Rw -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Cl -t Ar -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Cl -t Pr -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Cl -t Rw -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Pr -t Ar -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Pr -t Cl -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Pr -t Rw -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Rw -t Ar -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Rw -t Cl -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=0 python afn.py data/office-home -d OfficeHome -s Rw -t Pr -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/afn_vit/OfficeHome_Rw2Pr
