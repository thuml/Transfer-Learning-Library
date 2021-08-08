#!/usr/bin/env bash
# PACS
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P --freeze-bn --seed 0 --log logs/baseline/PACS_P
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A --freeze-bn --seed 0 --log logs/baseline/PACS_A
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C --freeze-bn --seed 0 --log logs/baseline/PACS_C
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S --freeze-bn --seed 0 --log logs/baseline/PACS_S
# Office-Home
CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr --seed 0 --log logs/baseline/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw --seed 0 --log logs/baseline/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl --seed 0 --log logs/baseline/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar --seed 0 --log logs/baseline/OfficeHome_Ar
# DomainNet
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s i p q r s -t c -i 2500 -b 40 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_c
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c p q r s -t i -i 2500 -b 40 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_i
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c i q r s -t p -i 2500 -b 40 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_p
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c i p r s -t q -i 2500 -b 40 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_q
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c i p q s -t r -i 2500 -b 40 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_r
CUDA_VISIBLE_DEVICES=0 python baseline.py data/domainnet -d DomainNet -s c i p q r -t s -i 2500 -b 40 --lr 0.01 --seed 0 --log logs/baseline/DomainNet_s
