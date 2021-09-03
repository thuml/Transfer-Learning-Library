#!/usr/bin/env bash
# Market1501 -> Duke
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/Market2DukeSeed0
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 1 --log logs/baseline/Market2DukeSeed1
# step2: train mmt
CUDA_VISIBLE_DEVICES=0,1,2,3 python mmt.py data data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--pretrained-model-1-path logs/baseline/Market2DukeSeed0/checkpoints/best.pth \
--pretrained-model-2-path logs/baseline/Market2DukeSeed1/checkpoints/best.pth \
--finetune --seed 0 --log logs/mmt/Market2Duke

# Duke -> Market1501
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/Duke2MarketSeed0
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 1 --log logs/baseline/Duke2MarketSeed1
# step2: train mmt
CUDA_VISIBLE_DEVICES=0,1,2,3 python mmt.py data data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--pretrained-model-1-path logs/baseline/Duke2MarketSeed0/checkpoints/best.pth \
--pretrained-model-2-path logs/baseline/Duke2MarketSeed1/checkpoints/best.pth \
--finetune --seed 0 --log logs/mmt/Duke2Market

# Market1501 -> MSMT
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s Market1501 -t MSMT17 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/Market2MSMTSeed0
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s Market1501 -t MSMT17 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 1 --log logs/baseline/Market2MSMTSeed1
# step2: train mmt
CUDA_VISIBLE_DEVICES=0,1,2,3 python mmt.py data data -s Market1501 -t MSMT17 -a reid_resnet50 \
--pretrained-model-1-path logs/baseline/Market2MSMTSeed0/checkpoints/best.pth \
--pretrained-model-2-path logs/baseline/Market2MSMTSeed1/checkpoints/best.pth \
--num-clusters 1000 --finetune --seed 0 --log logs/mmt/Market2MSMT

# MSMT -> Market1501
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s MSMT17 -t Market1501 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/MSMT2MarketSeed0
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s MSMT17 -t Market1501 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 1 --log logs/baseline/MSMT2MarketSeed1
# step2: train mmt
CUDA_VISIBLE_DEVICES=0,1,2,3 python mmt.py data data -s MSMT17 -t Market1501 -a reid_resnet50 \
--pretrained-model-1-path logs/baseline/MSMT2MarketSeed0/checkpoints/best.pth \
--pretrained-model-2-path logs/baseline/MSMT2MarketSeed1/checkpoints/best.pth \
--finetune --seed 0 --log logs/mmt/MSMT2Market

# Duke -> MSMT
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s DukeMTMC -t MSMT17 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/Duke2MSMTSeed0
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s DukeMTMC -t MSMT17 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 1 --log logs/baseline/Duke2MSMTSeed1
# step2: train mmt
CUDA_VISIBLE_DEVICES=0,1,2,3 python mmt.py data data -s DukeMTMC -t MSMT17 -a reid_resnet50 \
--pretrained-model-1-path logs/baseline/Duke2MSMTSeed0/checkpoints/best.pth \
--pretrained-model-2-path logs/baseline/Duke2MSMTSeed1/checkpoints/best.pth \
--num-clusters 1000 --finetune --seed 0 --log logs/mmt/Duke2MSMT

# MSMT -> Duke
# step1: pretrain
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s MSMT17 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/baseline/MSMT2DukeSeed0
CUDA_VISIBLE_DEVICES=0 python baseline.py data data -s MSMT17 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 1 --log logs/baseline/MSMT2DukeSeed1
# step2: train mmt
CUDA_VISIBLE_DEVICES=0,1,2,3 python mmt.py data data -s MSMT17 -t DukeMTMC -a reid_resnet50 \
--pretrained-model-1-path logs/baseline/MSMT2DukeSeed0/checkpoints/best.pth \
--pretrained-model-2-path logs/baseline/MSMT2DukeSeed1/checkpoints/best.pth \
--finetune --seed 0 --log logs/mmt/MSMT2Duke
