#!/usr/bin/env bash
# ResNet50, PACS
CUDA_VISIBLE_DEVICES=0 python irm.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 0 --log logs/irm/PACS_P
CUDA_VISIBLE_DEVICES=0 python irm.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 0 --log logs/irm/PACS_A
CUDA_VISIBLE_DEVICES=0 python irm.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 0 --log logs/irm/PACS_C
CUDA_VISIBLE_DEVICES=0 python irm.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 0 --log logs/irm/PACS_S

# ResNet50, Office-Home
CUDA_VISIBLE_DEVICES=0 python irm.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/irm/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=0 python irm.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --seed 0 --log logs/irm/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=0 python irm.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50 --seed 0 --log logs/irm/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=0 python irm.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50 --seed 0 --log logs/irm/OfficeHome_Ar

# ResNet50, DomainNet
CUDA_VISIBLE_DEVICES=0 python irm.py data/domainnet -d DomainNet -s i p q r s -t c -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --seed 0 --log logs/irm/DomainNet_c
CUDA_VISIBLE_DEVICES=0 python irm.py data/domainnet -d DomainNet -s c p q r s -t i -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --seed 0 --log logs/irm/DomainNet_i
CUDA_VISIBLE_DEVICES=0 python irm.py data/domainnet -d DomainNet -s c i q r s -t p -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --seed 0 --log logs/irm/DomainNet_p
CUDA_VISIBLE_DEVICES=0 python irm.py data/domainnet -d DomainNet -s c i p r s -t q -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --seed 0 --log logs/irm/DomainNet_q
CUDA_VISIBLE_DEVICES=0 python irm.py data/domainnet -d DomainNet -s c i p q s -t r -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --seed 0 --log logs/irm/DomainNet_r
CUDA_VISIBLE_DEVICES=0 python irm.py data/domainnet -d DomainNet -s c i p q r -t s -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --seed 0 --log logs/irm/DomainNet_s
