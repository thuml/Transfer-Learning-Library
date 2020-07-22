#!/usr/bin/env bash
# Office31
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 20  --bottleneck-dim 1024 --seed 1 > benchmarks/mdd/Office31_A2W.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office31 -d Office31 -s D -t W -a resnet50  --epochs 20  --bottleneck-dim 1024 --seed 1 > benchmarks/mdd/Office31_D2W.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office31 -d Office31 -s W -t D -a resnet50  --epochs 20  --bottleneck-dim 1024 --seed 1 > benchmarks/mdd/Office31_W2D.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office31 -d Office31 -s A -t D -a resnet50  --epochs 20  --bottleneck-dim 1024 --seed 1 > benchmarks/mdd/Office31_A2D.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office31 -d Office31 -s D -t A -a resnet50  --epochs 20  --bottleneck-dim 1024 --seed 1 > benchmarks/mdd/Office31_D2A.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office31 -d Office31 -s W -t A -a resnet50  --epochs 20  --bottleneck-dim 1024 --seed 1 > benchmarks/mdd/Office31_W2A.txt

# Office-Home
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Ar2Cl.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Ar2Pr.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  > benchmarks/mdd/OfficeHome_Ar2Rw.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50  --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Cl2Ar.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50  --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Cl2Pr.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50  --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Cl2Rw.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50  --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Pr2Ar.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50  --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Pr2Cl.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50  --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Pr2Rw.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50  --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Rw2Ar.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50  --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Rw2Cl.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50  --epochs 30 --bottleneck-dim 2048 --seed 0 > benchmarks/mdd/OfficeHome_Rw2Pr.txt

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet50  --epochs 30 --bottleneck-dim 1024 --seed 0 --center_crop > benchmarks/mdd/VisDA2017.txt
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101  --epochs 30 --bottleneck-dim 1024 --seed 0 --center_crop > benchmarks/mdd/VisDA2017_resnet101.txt

# DomainNet
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s c -t i -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s c -t p -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s c -t r -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s c -t s -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s i -t c -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s i -t p -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s i -t r -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s i -t s -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s p -t c -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s p -t i -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s p -t r -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s p -t s -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s r -t c -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s r -t i -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s r -t p -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s r -t s -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s s -t c -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s s -t i -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s s -t p -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
CUDA_VISIBLE_DEVICES=0 python examples/unsupervised-da/mdd.py data/domainnet -d DomainNet -s s -t r -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004
