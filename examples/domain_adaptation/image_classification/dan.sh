#!/usr/bin/env bash
# ResNet50, Office31, Single Source
CUDA_VISIBLE_DEVICES=0 python dan.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_D2A
CUDA_VISIBLE_DEVICES=0 python dan.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_W2A
CUDA_VISIBLE_DEVICES=0 python dan.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python dan.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_A2D
CUDA_VISIBLE_DEVICES=0 python dan.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_D2W
CUDA_VISIBLE_DEVICES=0 python dan.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_W2D

# ResNet50, Office-Home, Single Source
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 -i 500 --seed 0 --log logs/dan/OfficeHome_Rw2Pr

# ResNet101, VisDA-2017, Single Source
CUDA_VISIBLE_DEVICES=0 python dan.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 \
    --epochs 20 -i 500 --seed 0 --per-class-eval --train-resizing cen.crop --log logs/dan/VisDA2017

# ResNet101, DomainNet, Single Source
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s c -t p -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_c2p
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s c -t r -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_c2r
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s c -t s -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_c2s
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s p -t c -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_p2c
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s p -t r -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_p2r
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s p -t s -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_p2s
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s r -t c -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_r2c
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s r -t p -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_r2p
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s r -t s -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_r2s
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s s -t c -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_s2c
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s s -t p -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_s2p
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s s -t r -a resnet101 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/DomainNet_s2r

# ResNet50, ImageNet200 -> ImageNetR
CUDA_VISIBLE_DEVICES=0 python dan.py data/ImageNetR -d ImageNetR -s IN -t INR -a resnet50 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dan/ImageNet_IN2INR

# ig_resnext101_32x8d, ImageNet -> ImageNetSketch
CUDA_VISIBLE_DEVICES=0 python dan.py data/imagenet-sketch -d ImageNetSketch -s IN -t sketch -a ig_resnext101_32x8d --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --log logs/dan_ig_resnext101_32x8d/ImageNet_IN2sketch

# Vision Transformer, Office-Home, Single Source
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Ar -t Cl -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Ar -t Pr -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Ar -t Rw -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Cl -t Ar -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Cl -t Pr -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Cl -t Rw -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Pr -t Ar -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Pr -t Cl -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Pr -t Rw -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Rw -t Ar -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Rw -t Cl -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Rw -t Pr -a vit_base_patch16_224 --no-pool --epochs 30 --seed 0 -b 24 --log logs/dan_vit/OfficeHome_Rw2Pr

# ResNet50, Office-Home, Multi Source
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Cl Pr Rw -t Ar -a resnet50 --epochs 30 --seed 0 --log logs/dan/OfficeHome_:2Ar
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Ar Pr Rw -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/dan/OfficeHome_:2Cl
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --epochs 30 --seed 0 --log logs/dan/OfficeHome_:2Pr
CUDA_VISIBLE_DEVICES=0 python dan.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --epochs 30 --seed 0 --log logs/dan/OfficeHome_:2Rw

# ResNet101, DomainNet, Multi Source
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s i p q r s -t c -a resnet101 --bottleneck-dim 1024 --epochs 40 -i 5000 -p 500 --seed 0 --log logs/dan/DomainNet_:2c
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s c p q r s -t i -a resnet101 --bottleneck-dim 1024 --epochs 40 -i 5000 -p 500 --seed 0 --log logs/dan/DomainNet_:2i
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s c i q r s -t p -a resnet101 --bottleneck-dim 1024 --epochs 40 -i 5000 -p 500 --seed 0 --log logs/dan/DomainNet_:2p
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s c i p r s -t q -a resnet101 --bottleneck-dim 1024 --epochs 40 -i 5000 -p 500 --seed 0 --log logs/dan/DomainNet_:2q
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s c i p q s -t r -a resnet101 --bottleneck-dim 1024 --epochs 40 -i 5000 -p 500 --seed 0 --log logs/dan/DomainNet_:2r
CUDA_VISIBLE_DEVICES=0 python dan.py data/domainnet -d DomainNet -s c i p q r -t s -a resnet101 --bottleneck-dim 1024 --epochs 40 -i 5000 -p 500 --seed 0 --log logs/dan/DomainNet_:2s

# Digits
CUDA_VISIBLE_DEVICES=0 python dan.py data/digits -d Digits -s MNIST -t USPS --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 28 --no-hflip --norm-mean 0.5 --norm-std 0.5 -a lenet --no-pool --lr 0.01 -b 128 -i 2500 --scratch --seed 0 --log logs/dan/MNIST2USPS
CUDA_VISIBLE_DEVICES=0 python dan.py data/digits -d Digits -s USPS -t MNIST --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 28 --no-hflip --norm-mean 0.5 --norm-std 0.5 -a lenet --no-pool --lr 0.1 -b 128 -i 2500 --scratch --seed 0 --log logs/dan/USPS2MNIST
CUDA_VISIBLE_DEVICES=0 python dan.py data/digits -d Digits -s SVHNRGB -t MNISTRGB --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 32 --no-hflip --norm-mean 0.5 0.5 0.5 --norm-std 0.5 0.5 0.5 -a dtn --no-pool --lr 0.01 -b 128 -i 2500 --scratch --seed 0 --log logs/dan/SVHN2MNIST

