#!/usr/bin/env bash
# Office31
CUDA_VISIBLE_DEVICES=0 python se.py data/office31 -d Office31 -s A -t W -a resnet50 --seed 1 --log logs/se/Office31_A2W --pretrain logs/src_only/Office31_A2W/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office31 -d Office31 -s D -t W -a resnet50 --seed 1 --log logs/se/Office31_D2W --pretrain logs/src_only/Office31_D2W/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office31 -d Office31 -s W -t D -a resnet50 --seed 1 --log logs/se/Office31_W2D --pretrain logs/src_only/Office31_W2D/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office31 -d Office31 -s A -t D -a resnet50 --seed 1 --log logs/se/Office31_A2D --pretrain logs/src_only/Office31_A2D/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office31 -d Office31 -s D -t A -a resnet50 --seed 1 --log logs/se/Office31_D2A --pretrain logs/src_only/Office31_D2A/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office31 -d Office31 -s W -t A -a resnet50 --seed 1 --log logs/se/Office31_W2A --pretrain logs/src_only/Office31_W2A/checkpoints/best.pth

# Office-Home
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --seed 0 --log logs/se/OfficeHome_Ar2Cl --pretrain logs/src_only/OfficeHome_Ar2Cl/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --seed 0 --log logs/se/OfficeHome_Ar2Pr --pretrain logs/src_only/OfficeHome_Ar2Pr/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --seed 0 --log logs/se/OfficeHome_Ar2Rw --pretrain logs/src_only/OfficeHome_Ar2Rw/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --seed 0 --log logs/se/OfficeHome_Cl2Ar --pretrain logs/src_only/OfficeHome_Cl2Ar/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --seed 0 --log logs/se/OfficeHome_Cl2Pr --pretrain logs/src_only/OfficeHome_Cl2Pr/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --seed 0 --log logs/se/OfficeHome_Cl2Rw --pretrain logs/src_only/OfficeHome_Cl2Rw/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --seed 0 --log logs/se/OfficeHome_Pr2Ar --pretrain logs/src_only/OfficeHome_Pr2Ar/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --seed 0 --log logs/se/OfficeHome_Pr2Cl --pretrain logs/src_only/OfficeHome_Pr2Cl/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --seed 0 --log logs/se/OfficeHome_Pr2Rw --pretrain logs/src_only/OfficeHome_Pr2Rw/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --seed 0 --log logs/se/OfficeHome_Rw2Ar --pretrain logs/src_only/OfficeHome_Rw2Ar/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --seed 0 --log logs/se/OfficeHome_Rw2Cl --pretrain logs/src_only/OfficeHome_Rw2Cl/checkpoints/best.pth
CUDA_VISIBLE_DEVICES=0 python se.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --seed 0 --log logs/se/OfficeHome_Rw2Pr --pretrain logs/src_only/OfficeHome_Rw2Pr/checkpoints/best.pth

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python se.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 \
    --bottleneck-dim 512 --epochs 20 --seed 0 --per-class-eval --log logs/se/VisDA2017 --pretrain logs/src_only/VisDA2017/checkpoints/best.pth
