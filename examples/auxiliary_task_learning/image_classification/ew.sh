# Multi-Task Learning in a Equal-Weight fashion
CUDA_VISIBLE_DEVICES=0 python ew.py data/domainnet -d DomainNetv2 -tr c i p q r s -ts c i p q r s -a resnet101 \
  --epochs 20 --seed 0  --log logs/ew/DomainNet
