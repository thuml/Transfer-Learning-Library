# PCGrad
CUDA_VISIBLE_DEVICES=0 python pcgrad.py data/domainnet -d DomainNetv2 -tr c i p q r s -ts c i p q r s -a resnet101 \
  --epochs 20 --seed 0  --log logs/PCGrad_bs_48/DomainNet -b 48
