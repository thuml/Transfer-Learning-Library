#!/usr/bin/env bash
# ResNet50, Office31, Single Source
CUDA_VISIBLE_DEVICES=4 python cmu.py data/office31 -d Office31 -s D -t A -b 16 -i 20 -p 10 --epochs 50 --seed 0 --log logs/cmu/Office31_D2A
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 0 --log logs/cmu/Office31_W2A
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 0 --log logs/cmu/Office31_A2W
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 0 --log logs/cmu/Office31_A2D
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 0 --log logs/cmu/Office31_D2W
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 0 --log logs/cmu/Office31_W2D

# ResNet50, Office-Home, Single Source
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Ar2Cl
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Ar2Pr
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Ar2Rw
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Cl2Ar
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Cl2Pr
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Cl2Rw
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Pr2Ar
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Pr2Cl
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Pr2Rw
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Rw2Ar
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Rw2Cl
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/cmu/OfficeHome_Rw2Pr

# ResNet101, VisDA-2017, Single Source
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 \
#    --epochs 20 -i 500 --seed 0 --per-class-eval --train-resizing cen.crop --log logs/cmu/VisDA2017

# ResNet101, DomainNet, Single Source
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s c -t p -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_c2p
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s c -t r -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_c2r
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s c -t s -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_c2s
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s p -t c -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_p2c
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s p -t r -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_p2r
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s p -t s -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_p2s
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s r -t c -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_r2c
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s r -t p -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_r2p
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s r -t s -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_r2s
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s s -t c -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_s2c
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s s -t p -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_s2p
#CUDA_VISIBLE_DEVICES=0 python cmu.py data/domainnet -d DomainNet -s s -t r -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/cmu/DomainNet_s2r
