#!/usr/bin/env bash
# Office-Home
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/office-home -d OfficeHome -t Ar -a resnet50 --epochs 10 -i 1000 --seed 0 > benchmarks/da/multi_source/source_only/OfficeHome_:2Ar.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/office-home -d OfficeHome -t Cl -a resnet50 --epochs 10 -i 1000 --seed 0 > benchmarks/da/multi_source/source_only/OfficeHome_:2Cl.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/office-home -d OfficeHome -t Pr -a resnet50 --epochs 10 -i 1000 --seed 0 > benchmarks/da/multi_source/source_only/OfficeHome_:2Pr.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/office-home -d OfficeHome -t Rw -a resnet50 --epochs 10 -i 1000 --seed 0 > benchmarks/da/multi_source/source_only/OfficeHome_:2Rw.txt


# DomainNet
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/domainnet -d DomainNet -t c -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/da/multi_source/source_only/DomainNet_:2c.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/domainnet -d DomainNet -t i -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/da/multi_source/source_only/DomainNet_:2i.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/domainnet -d DomainNet -t p -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/da/multi_source/source_only/DomainNet_:2p.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/domainnet -d DomainNet -t q -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/da/multi_source/source_only/DomainNet_:2q.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/domainnet -d DomainNet -t r -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/da/multi_source/source_only/DomainNet_:2r.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/multi-source/source_only.py data/domainnet -d DomainNet -t s -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/da/multi_source/source_only/DomainNet_:2s.txt