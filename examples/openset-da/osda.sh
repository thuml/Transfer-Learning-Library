#!/usr/bin/env bash
# Office31
CUDA_VISIBLE_DEVICES=0 python examples/openset-da/osda.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 10 --seed 0 > benchmarks/osda/Office31_A2W.txt
CUDA_VISIBLE_DEVICES=0 python examples/openset-da/osda.py data/office31 -d Office31 -s D -t W -a resnet50  --epochs 10 --seed 0 > benchmarks/osda/Office31_D2W.txt
CUDA_VISIBLE_DEVICES=0 python examples/openset-da/osda.py data/office31 -d Office31 -s W -t D -a resnet50  --epochs 10 --seed 0 > benchmarks/osda/Office31_W2D.txt
CUDA_VISIBLE_DEVICES=0 python examples/openset-da/osda.py data/office31 -d Office31 -s A -t D -a resnet50  --epochs 10 --seed 0 > benchmarks/osda/Office31_A2D.txt
CUDA_VISIBLE_DEVICES=0 python examples/openset-da/osda.py data/office31 -d Office31 -s D -t A -a resnet50  --epochs 10 --seed 0 > benchmarks/osda/Office31_D2A.txt
CUDA_VISIBLE_DEVICES=0 python examples/openset-da/osda.py data/office31 -d Office31 -s W -t A -a resnet50  --epochs 10 --seed 0 > benchmarks/osda/Office31_W2A.txt

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python examples/openset-da/osda.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet50  --epochs 20 -i 1000 --seed 0  > benchmarks/osda/VisDA2017_S2R.txt