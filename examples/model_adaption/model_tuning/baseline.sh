#!/usr/bin/env bash

# ======================================================================================================================
# FGVCAircraft

CUDA_VISIBLE_DEVICES=0 python baseline.py data/FGVCAircraft -d Aircraft --arch resnet50 \
    --seed 0 --lr 0.01 --finetune --log logs/baseline/Aircraft_resnet50 

CUDA_VISIBLE_DEVICES=3 python baseline.py data/caltech101 -d Caltech101 --arch resnet50 \
    --seed 0 --lr 0.01 --finetune --log logs/baseline/Caltech101_resnet50 