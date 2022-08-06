#!/usr/bin/env bash
# ResNet50, Office31, Single Source
CUDA_VISIBLE_DEVICES=0 python cmu.py data/office31 -d Office31 -s D -t A --threshold 0.5 --src-threshold 0.25 --cut 0.1 --seed 0 --log logs/cmu/Office31_D2A

# ResNet50, Office-Home, Single Source

# ResNet101, VisDA-2017, Single Source

# ResNet101, DomainNet, Single Source
