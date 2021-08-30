#!/usr/bin/env bash
# Market1501 -> Duke
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data -s Market1501 -t DukeMTMC -a resnet50 \
--mix-layers layer1 layer2 --finetune --seed 0 --log logs/mixstyle/Market2Duke

# Duke -> Market1501
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data -s DukeMTMC -t Market1501 -a resnet50 \
--mix-layers layer1 layer2 --finetune --seed 0 --log logs/mixstyle/Duke2Market

# Market1501 -> MSMT
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data -s Market1501 -t MSMT17 -a resnet50 \
--mix-layers layer1 layer2 --finetune --seed 0 --log logs/mixstyle/Market2MSMT

# MSMT -> Market1501
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data -s MSMT17 -t Market1501 -a resnet50 \
--mix-layers layer1 layer2 --finetune --seed 0 --log logs/mixstyle/MSMT2Market

# Duke -> MSMT
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data -s DukeMTMC -t MSMT17 -a resnet50 \
--mix-layers layer1 layer2 --finetune --seed 0 --log logs/mixstyle/Duke2MSMT

# MSMT -> Duke
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data -s MSMT17 -t DukeMTMC -a resnet50 \
--mix-layers layer1 layer2 --finetune --seed 0 --log logs/mixstyle/MSMT2Duke
