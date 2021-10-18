#!/usr/bin/env bash
# Market1501 -> Duke
CUDA_VISIBLE_DEVICES=0,1,2,3 python idm.py data data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--seed 0 --log logs/idm/Market2Duke

# Duke -> Market1501
CUDA_VISIBLE_DEVICES=0,1,2,3 python idm.py data data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--seed 0 --log logs/idm/Duke2Market

# Market1501 -> MSMT
CUDA_VISIBLE_DEVICES=0,1,2,3 python idm.py data data -s Market1501 -t MSMT17 -a reid_resnet50 \
--seed 0 --log logs/idm/Market2MSMT

# MSMT -> Market1501
CUDA_VISIBLE_DEVICES=0,1,2,3 python idm.py data data -s MSMT17 -t Market1501 -a reid_resnet50 \
--seed 0 --log logs/idm/MSMT2Market

# Duke -> MSMT
CUDA_VISIBLE_DEVICES=0,1,2,3 python idm.py data data -s DukeMTMC -t MSMT17 -a reid_resnet50 \
--seed 1 --log logs/idm/Duke2MSMT

# MSMT -> Duke
CUDA_VISIBLE_DEVICES=0,1,2,3 python idm.py data data -s MSMT17 -t DukeMTMC -a reid_resnet50 \
--seed 0 --log logs/idm/MSMT2Duke
