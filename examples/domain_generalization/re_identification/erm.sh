#!/usr/bin/env bash
# Market1501 -> Duke
CUDA_VISIBLE_DEVICES=0 python erm.py data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--finetune --seed 0 --log logs/erm/Market2Duke

# Duke -> Market1501
CUDA_VISIBLE_DEVICES=0 python erm.py data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--finetune --seed 0 --log logs/erm/Duke2Market

# Market1501 -> MSMT
CUDA_VISIBLE_DEVICES=0 python erm.py data -s Market1501 -t MSMT17 -a reid_resnet50 \
--finetune --seed 0 --log logs/erm/Market2MSMT

# MSMT -> Market1501
CUDA_VISIBLE_DEVICES=0 python erm.py data -s MSMT17 -t Market1501 -a reid_resnet50 \
--finetune --seed 0 --log logs/erm/MSMT2Market

# Duke -> MSMT
CUDA_VISIBLE_DEVICES=0 python erm.py data -s DukeMTMC -t MSMT17 -a reid_resnet50 \
--finetune --seed 0 --log logs/erm/Duke2MSMT

# MSMT -> Duke
CUDA_VISIBLE_DEVICES=0 python erm.py data -s MSMT17 -t DukeMTMC -a reid_resnet50 \
--finetune --seed 0 --log logs/erm/MSMT2Duke
