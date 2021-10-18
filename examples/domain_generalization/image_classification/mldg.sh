#!/usr/bin/env bash
# ResNet50, PACS
CUDA_VISIBLE_DEVICES=0 python mldg.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 0 --log logs/mldg/PACS_P
CUDA_VISIBLE_DEVICES=0 python mldg.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 0 --log logs/mldg/PACS_A
CUDA_VISIBLE_DEVICES=0 python mldg.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 0 --log logs/mldg/PACS_C
CUDA_VISIBLE_DEVICES=0 python mldg.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 0 --log logs/mldg/PACS_S

# ResNet50, Office-Home
CUDA_VISIBLE_DEVICES=0 python mldg.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/mldg/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=0 python mldg.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --seed 0 --log logs/mldg/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=0 python mldg.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50 --seed 0 --log logs/mldg/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=0 python mldg.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50 --seed 0 --log logs/mldg/OfficeHome_Ar

# ResNet50, DomainNet
CUDA_VISIBLE_DEVICES=0 python mldg.py data/domainnet -d DomainNet -s i p q r s -t c -a resnet50 -i 5000 --lr 0.005 --seed 0 --log logs/mldg/DomainNet_c
CUDA_VISIBLE_DEVICES=0 python mldg.py data/domainnet -d DomainNet -s c p q r s -t i -a resnet50 -i 5000 --lr 0.005 --seed 0 --log logs/mldg/DomainNet_i
CUDA_VISIBLE_DEVICES=0 python mldg.py data/domainnet -d DomainNet -s c i q r s -t p -a resnet50 -i 5000 --lr 0.005 --seed 0 --log logs/mldg/DomainNet_p
CUDA_VISIBLE_DEVICES=0 python mldg.py data/domainnet -d DomainNet -s c i p r s -t q -a resnet50 -i 5000 --lr 0.005 --seed 0 --log logs/mldg/DomainNet_q
CUDA_VISIBLE_DEVICES=0 python mldg.py data/domainnet -d DomainNet -s c i p q s -t r -a resnet50 -i 5000 --lr 0.005 --seed 0 --log logs/mldg/DomainNet_r
CUDA_VISIBLE_DEVICES=0 python mldg.py data/domainnet -d DomainNet -s c i p q r -t s -a resnet50 -i 5000 --lr 0.005 --seed 0 --log logs/mldg/DomainNet_s

# ResNet50, Wilds Dataset
CUDA_VISIBLE_DEVICES=0 python mldg.py data/wilds -d camelyon17 -a resnet50 -b 36 \
  --n-support-domains 1 --n-query-domains 2 --epochs 20 -i 1000 --lr 0.003 --seed 0 --log logs/mldg/camelyon17

# DenseNet121, Wilds Dataset
CUDA_VISIBLE_DEVICES=5 python mldg.py data/wilds -d fmow --train-resizing 'cen.crop' -a densenet121 \
  -b 36 --n-support-domains 2 --n-query-domains 2 --epochs 60 -i 1000 --lr 0.001 --seed 0 --log logs/mldg/fmow
