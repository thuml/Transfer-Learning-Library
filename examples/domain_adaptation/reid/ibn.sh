#!/usr/bin/env bash
# Market1501 -> Duke
# ibn_a
CUDA_VISIBLE_DEVICES=4 python baseline.py data -s Market1501 -t DukeMTMC -a resnet50_ibn_a \
--seed 0 --log logs/baseline/Market2Duke --finetune
# ibn_b
CUDA_VISIBLE_DEVICES=4 python baseline.py data -s Market1501 -t DukeMTMC -a resnet50_ibn_b \
--seed 0 --log logs/baseline/Market2Duke --finetune
# use command below to train longer and achieve higher mAP
CUDA_VISIBLE_DEVICES=1 python baseline.py data -s Market1501 -t DukeMTMC -a resnet50_ibn_b \
--iters-per-epoch 1000 --print-freq 100 --seed 0 --log logs/baseline/Market2Duke --finetune
# Duke -> Market1501
# ibn_a
CUDA_VISIBLE_DEVICES=5 python baseline.py data -s DukeMTMC -t Market1501 -a resnet50_ibn_a \
--seed 0 --log logs/baseline/Duke2Market --finetune
# ibn_b
CUDA_VISIBLE_DEVICES=5 python baseline.py data -s DukeMTMC -t Market1501 -a resnet50_ibn_b \
--seed 0 --log logs/baseline/Duke2Market --finetune
# use command below to train longer and achieve higher mAP
CUDA_VISIBLE_DEVICES=3 python baseline.py data -s DukeMTMC -t Market1501 -a resnet50_ibn_b \
--iters-per-epoch 1000 --print-freq 100 --seed 0 --log logs/baseline/Duke2Market --finetune
