#!/usr/bin/env bash
# Office31
CUDA_VISIBLE_DEVICES=0 python erm.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 5 --seed 0 --log logs/erm/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python erm.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 5 --seed 0 --log logs/erm/Office31_D2W
CUDA_VISIBLE_DEVICES=0 python erm.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 5 --seed 0 --log logs/erm/Office31_W2D
CUDA_VISIBLE_DEVICES=0 python erm.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 5 --seed 0 --log logs/erm/Office31_A2D
CUDA_VISIBLE_DEVICES=0 python erm.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 5 --seed 0 --log logs/erm/Office31_D2A
CUDA_VISIBLE_DEVICES=0 python erm.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 5 --seed 0 --log logs/erm/Office31_W2A

# Office-Home
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/erm/OfficeHome_Rw2Pr

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python erm.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet50 \
    --epochs 10 -i 500 --seed 0 --per-class-eval --log logs/erm/VisDA2017_S2R

# ImageNet-Caltech
CUDA_VISIBLE_DEVICES=0 python erm.py data/ImageNetCaltech -d ImageNetCaltech -s I -t C -a resnet50 \
    --epochs 20 --seed 0 -i 2000 --log logs/erm/I2C
CUDA_VISIBLE_DEVICES=0 python erm.py data/ImageNetCaltech -d CaltechImageNet -s C -t I -a resnet50 \
    --epochs 20 --seed 0 -i 2000 --log logs/erm/C2I
