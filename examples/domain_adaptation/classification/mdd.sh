#!/usr/bin/env bash
# Office31
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_D2W
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_W2D
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_A2D
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_D2A
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_W2A

# Office-Home
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Rw2Pr

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python mdd.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --epochs 30 \
    --bottleneck-dim 1024 --seed 0 --train-resizing cen.crop --per-class-eval -b 36 --log logs/mdd/VisDA2017

# DomainNet
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s c -t i -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_c2i
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s c -t p -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_c2p
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s c -t r -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_c2r
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s c -t s -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_c2s
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s i -t c -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_i2c
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s i -t p -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_i2p
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s i -t r -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_i2r
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s i -t s -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_i2s
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s p -t c -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_p2c
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s p -t i -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_p2i
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s p -t r -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_p2r
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s p -t s -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_p2s
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s r -t c -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_r2c
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s r -t i -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_r2i
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s r -t p -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_r2p
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s r -t s -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_r2s
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s s -t c -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_s2c
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s s -t i -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_s2i
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s s -t p -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_s2p
CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -s s -t r -a resnet101 --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_s2r

# Wilds Dataset
CUDA_VISIBLE_DEVICES=0 python mdd.py data/wilds -d iwildcam -a resnet50 --epochs 40 -i 1000 --bottleneck-dim 2048 --seed 0 --log logs/mdd/iwildcam
CUDA_VISIBLE_DEVICES=0 python mdd.py data/wilds -d fmow -a resnet50 --epochs 40 -i 1000 --bottleneck-dim 2048 --seed 0 --log logs/mdd/fmow

# Office-Home on Vision Transformer
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Ar -t Cl -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Ar -t Pr -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Ar -t Rw -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Cl -t Ar -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Cl -t Pr -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Cl -t Rw -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Pr -t Ar -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Pr -t Cl -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Pr -t Rw -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Rw -t Ar -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -s Rw -t Cl -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=6 python mdd.py data/office-home -d OfficeHome -s Rw -t Pr -a vit_base_patch16_224 --epochs 40 --bottleneck-dim 2048 --seed 0 -b 24 --no-pool --log logs/mdd_vit/OfficeHome_Rw2Pr