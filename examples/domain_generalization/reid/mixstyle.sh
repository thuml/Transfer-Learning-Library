#!/usr/bin/env bash
# Market1501 -> Duke
CUDA_VISIBLE_DEVICES=6 python mixstyle.py data -s Market1501 -t DukeMTMC -a resnet50 \
--mix-layers layer1 layer2 --seed 0 --log logs/mixstyle/Market2Duke --finetune
# Duke -> Market1501
CUDA_VISIBLE_DEVICES=7 python mixstyle.py data -s DukeMTMC -t Market1501 -a resnet50 \
--mix-layers layer1 layer2 --seed 0 --log logs/mixstyle/Duke2Market --finetune
