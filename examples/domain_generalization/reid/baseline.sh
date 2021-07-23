#!/usr/bin/env bash
# Market1501 -> Duke
CUDA_VISIBLE_DEVICES=3 python baseline.py data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--seed 0 --log logs/baseline/Market2Duke --finetune
# test
CUDA_VISIBLE_DEVICES=4 python baseline.py data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--log logs/baseline/Market2Duke --phase test
# Duke -> Market1501
CUDA_VISIBLE_DEVICES=3 python baseline.py data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--seed 0 --log logs/baseline/Duke2Market --finetune
# test
CUDA_VISIBLE_DEVICES=5 python baseline.py data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--log logs/baseline/Duke2Market --phase test
