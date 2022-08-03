#!/usr/bin/env bash
# ResNet50, Office31, Single Source
CUDA_VISIBLE_DEVICES=4 python cmu.py data/office31 -d Office31 -s D -t A -b 16 -i 20 --epochs 50 --seed 0 --log logs/cmu/Office31_D2A
CUDA_VISIBLE_DEVICES=4 python cmu.py data/office31 -d Office31 -s D -t A -i 200 --threshold 0.5 --src-threshold 0.25 --cut 0.1 --seed 0 --log logs/cmu/Office31_D2A
#CUDA_VISIBLE_DEVICES=4 python cmu.py data/office31 -d Office31 -s A -t D -b 16 -i 100 --epochs 50 --seed 0 --log logs/cmu/Office31_A2D
#CUDA_VISIBLE_DEVICES=5 python cmu.py data/office31 -d Office31 -s W -t A -b 16 -i 40 --epochs 50 --seed 0 --log logs/cmu/Office31_W2A


# ResNet50, Office-Home, Single Source


# ResNet101, VisDA-2017, Single Source


# ResNet101, DomainNet, Single Source

