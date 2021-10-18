#!/usr/bin/env bash
# ResNet50, PACS
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_P
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_A
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_C
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_S

# ResNet50, Office-Home
CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Ar

# ResNet50, DomainNet
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s i p q r s -t c -a resnet50 -i 2500 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_c
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c p q r s -t i -a resnet50 -i 2500 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_i
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c i q r s -t p -a resnet50 -i 2500 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_p
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c i p r s -t q -a resnet50 -i 2500 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_q
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c i p q s -t r -a resnet50 -i 2500 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_r
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c i p q r -t s -a resnet50 -i 2500 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_s

# ResNet50, Wilds Dataset
CUDA_VISIBLE_DEVICES=0 python baseline.py data/wilds -d iwildcam --train-resizing 'res2x' --val-resizing 'res2x' \
  -a resnet50 -b 16 --epochs 60 -i 1000 --lr 0.001 --finetune --seed 0 --log logs/baseline/iwildcam
CUDA_VISIBLE_DEVICES=0 python baseline.py data/wilds -d camelyon17 -a resnet50 -b 36 --epochs 20 -i 1000 --lr 0.01 \
  --seed 0 --log logs/baseline/camelyon17

# DenseNet121, Wilds Dataset
CUDA_VISIBLE_DEVICES=0 python baseline.py data/wilds -d fmow --train-resizing 'cen.crop' -a densenet121 \
  -b 36 --epochs 60 -i 1000 --lr 0.003 --seed 0 --log logs/baseline/fmow
