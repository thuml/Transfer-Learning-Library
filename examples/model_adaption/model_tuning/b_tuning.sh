#!/usr/bin/env bash

# ======================================================================================================================
# FGVCAircraft

# Ranking student and teacher models by LogME first
# cd ../model_selection
# CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a resnet50 -l fc --save_features --save_distribution
# CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a resnet101 -l fc --save_features --save_distribution
# CUDA_VISIBLE_DEVICES=0 python logme.py ./data/FGVCAircraft -d Aircraft -a resnet152 -l fc --save_features --save_distribution

CUDA_VISIBLE_DEVICES=2 python b_tuning.py data/FGVCAircraft -d Aircraft --seed 0 --arch resnet50 -t resnet101 resnet152 inception_v3\
    --lr 0.01 --result-path ../model_selection/logs/Aircraft --trade-off 1000 --finetune --log logs/btuning/Aircraft_S_resnet50_T_resnet101_resnet152_inception_v3_tf_1000

CUDA_VISIBLE_DEVICES=4 python b_tuning.py data/caltech101 -d Caltech101 --seed 0 --arch resnet50 -t resnet101 resnet152 inception_v3 \
    --lr 0.01 --result-path ../model_selection/logs/Caltech101 --trade-off 1000 --finetune --log logs/btuning/Caltech101_S_resnet50_T_resnet101_resnet152_inception_v3_tf_1000 &