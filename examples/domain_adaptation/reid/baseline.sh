#!/usr/bin/env bash
# Market1501 -> Duke
CUDA_VISIBLE_DEVICES=0 python baseline.py data -s Market1501 -t DukeMTMC -a resnet50 \
--seed 0 --log logs/baseline/Market2Duke
CUDA_VISIBLE_DEVICES=0,1,6,7 python baseline.py data -s Market1501 -t DukeMTMC -a resnet50 \
-b 64 --iters_per_epoch 100 --print-freq 10 --seed 0 --log logs/baselinebatch64seed0/Market2Duke

CUDA_VISIBLE_DEVICES=0 python baseline.py data -s Market1501 -t DukeMTMC -a resnet50 \
--log logs/baseline/Market2Duke --phase test
# Duke -> Market1501
CUDA_VISIBLE_DEVICES=1 python baseline.py data -s DukeMTMC -t Market1501 -a resnet50 \
--seed 0 --log logs/baseline/Duke2Market
CUDA_VISIBLE_DEVICES=0,1,6,7 python baseline.py data -s DukeMTMC -t Market1501 -a resnet50 \
-b 64 --iters_per_epoch 100 --print-freq 10 --seed 0 --log logs/baselinebatch64seed0/Duke2Market

CUDA_VISIBLE_DEVICES=1 python baseline.py data -s DukeMTMC -t Market1501 -a resnet50 \
--log logs/baseline/Duke2Market --phase test