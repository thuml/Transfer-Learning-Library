#!/usr/bin/env bash

# ======================================================================================================================
# FGVCAircraft

CUDA_VISIBLE_DEVICES=1 python zoo_tuning.py data/FGVCAircraft -d Aircraft --archs imageNet moco maskrcnn deeplab keyPoint \
    --seed 0 --lr 0.01 --log logs/zoo_tuning/Aircraft_imageNet_moco_maskrcnn_deeplab_keyPoint

CUDA_VISIBLE_DEVICES=2 python zoo_tuning.py data/FGVCAircraft -d Aircraft --archs imageNet moco maskrcnn deeplab keyPoint \
    --seed 0 --lite --lr 0.01 --log logs/zoo_tuning/Aircraft_lite_imageNet_moco_maskrcnn_deeplab_keyPoint