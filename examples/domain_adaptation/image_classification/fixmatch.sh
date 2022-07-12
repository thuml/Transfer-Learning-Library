#!/usr/bin/env bash
# ResNet50, Office31, Single Source
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office31 -d Office31 -s A -t W -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 20 --seed 0 --log logs/fixmatch/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office31 -d Office31 -s D -t W -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 20 --seed 0 --log logs/fixmatch/Office31_D2W
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office31 -d Office31 -s W -t D -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 20 --seed 0 --log logs/fixmatch/Office31_W2D
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office31 -d Office31 -s A -t D -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 20 --seed 0 --log logs/fixmatch/Office31_A2D
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office31 -d Office31 -s D -t A -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 20 --seed 0 --log logs/fixmatch/Office31_D2A
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office31 -d Office31 -s W -t A -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 20 --seed 0 --log logs/fixmatch/Office31_W2A

# ResNet50, Office-Home, Single Source
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --lr 0.003 --bottleneck-dim 1024 --epochs 20 --seed 0 --log logs/fixmatch/OfficeHome_Rw2Pr

# ResNet101, VisDA-2017, Single Source
CUDA_VISIBLE_DEVICES=0 python fixmatch.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --train-resizing cen.crop \
    --lr 0.003 --threshold 0.8 --bottleneck-dim 2048 --epochs 20 -ub 64 --seed 0 --per-class-eval --log logs/fixmatch/VisDA2017
