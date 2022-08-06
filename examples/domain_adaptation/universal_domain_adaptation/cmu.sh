#!/usr/bin/env bash
# ResNet50, Office31, Single Source
CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s A -t W --threshold 0.5 --src-threshold 0.25 --cut 0.1 --seed 0 --log logs/cmu/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s D -t W --threshold 0.5 --src-threshold 0.25 --cut 0.1 --seed 0 --log logs/cmu/Office31_D2W
CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s W -t D --threshold 0.5 --src-threshold 0.25 --cut 0.1 --seed 0 --log logs/cmu/Office31_W2D
CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s A -t D --threshold 0.5 --src-threshold 0.25 --cut 0.1 --seed 0 --log logs/cmu/Office31_A2D
CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s D -t A --threshold 0.5 --src-threshold 0.25 --cut 0.1 --seed 0 --log logs/cmu/Office31_D2A
CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s W -t A --threshold 0.5 --src-threshold 0.25 --cut 0.1 --seed 0 --log logs/cmu/Office31_W2A
