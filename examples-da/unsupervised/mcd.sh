#!/usr/bin/env bash
# Office31
# We found MCD loss is sensitive to class number,
# thus, when the class number increase, please increase trade-off correspondingly.
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 0 -i 500 --trade-off 10.0 --log logs/mcd/Office31_D2A
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 0 -i 500 --trade-off 10.0 --log logs/mcd/Office31_W2A
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 0 -i 500 --trade-off 10.0 --log logs/mcd/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 0 -i 500 --trade-off 10.0 --log logs/mcd/Office31_A2D
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 0 -i 500 --trade-off 10.0 --log logs/mcd/Office31_D2W
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 0 -i 500 --trade-off 10.0 --log logs/mcd/Office31_W2D

# Office-Home
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=0 python mcd.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 -i 500 --seed 0 --trade-off 30.0 --log logs/mcd/OfficeHome_Rw2Pr

# VisDA2017
CUDA_VISIBLE_DEVICES=0 python mcd.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 \
    --epochs 20 --center-crop --seed 0 -i 500 --per-class-eval --log logs/mcd/VisDA2017

