#!/usr/bin/env bash
# Office-Home
CUDA_VISIBLE_DEVICES=5 python source_only.py data/office-home -d OfficeHome -t Ar -a resnet50 --epochs 10 -i 1000 --seed 0 --log logs/src_only/OfficeHome_:2Ar
CUDA_VISIBLE_DEVICES=0 python source_only.py data/office-home -d OfficeHome -t Cl -a resnet50 --epochs 10 -i 1000 --seed 0 --log logs/src_only/OfficeHome_:2Cl
CUDA_VISIBLE_DEVICES=0 python source_only.py data/office-home -d OfficeHome -t Pr -a resnet50 --epochs 10 -i 1000 --seed 0 --log logs/src_only/OfficeHome_:2Pr
CUDA_VISIBLE_DEVICES=0 python source_only.py data/office-home -d OfficeHome -t Rw -a resnet50 --epochs 10 -i 1000 --seed 0 --log logs/src_only/OfficeHome_:2Rw


# DomainNet
CUDA_VISIBLE_DEVICES=0 python source_only.py data/domainnet -d DomainNet -t c -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 --log logs/src_only/DomainNet_:2c
CUDA_VISIBLE_DEVICES=0 python source_only.py data/domainnet -d DomainNet -t i -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 --log logs/src_only/DomainNet_:2i
CUDA_VISIBLE_DEVICES=0 python source_only.py data/domainnet -d DomainNet -t p -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 --log logs/src_only/DomainNet_:2p
CUDA_VISIBLE_DEVICES=0 python source_only.py data/domainnet -d DomainNet -t q -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 --log logs/src_only/DomainNet_:2q
CUDA_VISIBLE_DEVICES=0 python source_only.py data/domainnet -d DomainNet -t r -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 --log logs/src_only/DomainNet_:2r
CUDA_VISIBLE_DEVICES=0 python source_only.py data/domainnet -d DomainNet -t s -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 --log logs/src_only/DomainNet_:2s