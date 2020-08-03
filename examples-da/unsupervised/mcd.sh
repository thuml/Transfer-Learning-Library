#!/usr/bin/env bash
# VisDA2017
CUDA_VISIBLE_DEVICES=0 python examples-da/unsupervised/mcd.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet50  --epochs 20 --center-crop --seed 0 -i 500 > benchmarks/mcd/VisDA2017.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/unsupervised/mcd.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101  --epochs 20 --center-crop --seed 0 -i 500 > benchmarks/mcd/VisDA2017_resnet101.txt