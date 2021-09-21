#!/usr/bin/env bash
# Market1501 -> Duke
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/Market2Duke
# step2: train with pseudo labels assigned by cluster algorithm
CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline_cluster.py data data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--pretrained-model-path logs/baseline/Market2Duke/checkpoints/best.pth \
--finetune --seed 0 --log logs/baseline_cluster/Market2Duke

# Duke -> Market1501
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/Duke2Market
# step2: train with pseudo labels assigned by cluster algorithm
CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline_cluster.py data data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--pretrained-model-path logs/baseline/Duke2Market/checkpoints/best.pth \
--finetune --seed 0 --log logs/baseline_cluster/Duke2Market

# Market1501 -> MSMT
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s Market1501 -t MSMT17 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/Market2MSMT
# step2: train with pseudo labels assigned by cluster algorithm
CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline_cluster.py data data -s Market1501 -t MSMT17 -a reid_resnet50 \
--pretrained-model-path logs/baseline/Market2MSMT/checkpoints/best.pth \
--num-clusters 1000 --finetune --seed 0 --log logs/baseline_cluster/Market2MSMT

# MSMT -> Market1501
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s MSMT17 -t Market1501 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/MSMT2Market
# step2: train with pseudo labels assigned by cluster algorithm
CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline_cluster.py data data -s MSMT17 -t Market1501 -a reid_resnet50 \
--pretrained-model-path logs/baseline/MSMT2Market/checkpoints/best.pth \
--finetune --seed 0 --log logs/baseline_cluster/MSMT2Market

# Duke -> MSMT
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s DukeMTMC -t MSMT17 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/Duke2MSMT
# step2: train with pseudo labels assigned by cluster algorithm
CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline_cluster.py data data -s DukeMTMC -t MSMT17 -a reid_resnet50 \
--pretrained-model-path logs/baseline/Duke2MSMT/checkpoints/best.pth \
--num-clusters 1000 --finetune --seed 0 --log logs/baseline_cluster/Duke2MSMT

# MSMT -> Duke
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s MSMT17 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/MSMT2Duke
# step2: train with pseudo labels assigned by cluster algorithm
CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline_cluster.py data data -s MSMT17 -t DukeMTMC -a reid_resnet50 \
--pretrained-model-path logs/baseline/MSMT2Duke/checkpoints/best.pth \
--finetune --seed 0 --log logs/baseline_cluster/MSMT2Duke
