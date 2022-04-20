#!/usr/bin/env bash

# ======================================================================================================================
# FGVCAircraft

CUDA_VISIBLE_DEVICES=0 python distill.py data/FGVCAircraft -d Aircraft --seed 0 --arch resnet50 -t resnet101 resnet152 inception_v3 \
    --lr 0.01 --finetune --trade-off 3e-2 --log logs/distill/Aircraft_S_resnet50_T_resnet101_resnet152_inception_v3_tf_3e-2


CUDA_VISIBLE_DEVICES=5 python distill.py data/caltech101 -d Caltech101 --seed 0 --arch resnet50 -t resnet101 resnet152 inception_v3 \
    --lr 0.01 --finetune --trade-off 3e-2 --log logs/distill/Caltech101_S_resnet50_T_resnet101_resnet152_inception_v3_tf_3e-2 &