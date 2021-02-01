#!/usr/bin/env bash
# Office-Home
CUDA_VISIBLE_DEVICES=1 python examples-da/multi-source/mdd.py data/office-home -d OfficeHome -t Ar -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/da/multi_source/mdd/OfficeHome_:2Ar.txt
CUDA_VISIBLE_DEVICES=1 python examples-da/multi-source/mdd.py data/office-home -d OfficeHome -t Cl -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/da/multi_source/mdd/OfficeHome_:2Cl.txt
CUDA_VISIBLE_DEVICES=1 python examples-da/multi-source/mdd.py data/office-home -d OfficeHome -t Pr -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/da/multi_source/mdd/OfficeHome_:2Pr.txt
CUDA_VISIBLE_DEVICES=1 python examples-da/multi-source/mdd.py data/office-home -d OfficeHome -t Rw -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/da/multi_source/mdd/OfficeHome_:2Rw.txt

# DomainNet
CUDA_VISIBLE_DEVICES=3 python examples-da/multi-source/mdd.py data/domainnet -d DomainNet -t c -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 > benchmarks/da/multi_source/mdd/DomainNet_:2c.txt
CUDA_VISIBLE_DEVICES=4 python examples-da/multi-source/mdd.py data/domainnet -d DomainNet -t i -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 > benchmarks/da/multi_source/mdd/DomainNet_:2i.txt
CUDA_VISIBLE_DEVICES=3 python examples-da/multi-source/mdd.py data/domainnet -d DomainNet -t p -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 > benchmarks/da/multi_source/mdd/DomainNet_:2p.txt
CUDA_VISIBLE_DEVICES=3 python examples-da/multi-source/mdd.py data/domainnet -d DomainNet -t q -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 > benchmarks/da/multi_source/mdd/DomainNet_:2q.txt
CUDA_VISIBLE_DEVICES=3 python examples-da/multi-source/mdd.py data/domainnet -d DomainNet -t r -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 > benchmarks/da/multi_source/mdd/DomainNet_:2r.txt
CUDA_VISIBLE_DEVICES=3 python examples-da/multi-source/mdd.py data/domainnet -d DomainNet -t s -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 > benchmarks/da/multi_source/mdd/DomainNet_:2s.txt


CUDA_VISIBLE_DEVICES=1 python examples-da/multi-source/mdd.py data/domainnet -d DomainNet -t c -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 -b 36